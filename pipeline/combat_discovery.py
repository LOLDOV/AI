"""Full-VOD storyboard scouting + optional Gemini dense verification.

An experimental *candidate discovery* workflow, NOT a validated coaching-moment detector.
No human labels are read. No previous scene results are used. No data deletion.

- Each non-overlapping window is scanned (all the way to the end).
- Scouting batches are cached and resumable.
- Top candidates receive a second, denser VLM inspection.
- Results are ranked across the ENTIRE video before choosing top K.
- Explicit permission (--execute) is required for Gemini API calls.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import ROOT, dump, load, file_fingerprint, safe_id, timecode, video_meta

PROMPT_VERSION = 'coaching_discovery_storyboard_v4.0'
# Candidate discovery only. Importance is a VLM estimate, not ground truth.
CATEGORIES = {
    'champion_fight', 'objective_fight', 'possible_champion_fight',
    'lane_trade', 'skill_dodge', 'positioning', 'escape', 'macro_decision',
    'lane_farming', 'jungle_farming', 'movement', 'death_respawn',
    'recall_shop', 'unknown',
}
# Small category priors keep fights useful without automatically dominating coaching scenes.
BASE_SCORE = {
    'champion_fight': 2, 'objective_fight': 3,
    'possible_champion_fight': 1, 'lane_trade': 1,
    'skill_dodge': 2, 'positioning': 2, 'escape': 2,
    'macro_decision': 1, 'lane_farming': 0, 'jungle_farming': 0,
    'movement': 0, 'death_respawn': 0, 'recall_shop': 0, 'unknown': 0,
}
FIGHT_CATEGORIES = {'champion_fight', 'objective_fight', 'possible_champion_fight'}
COACH_CATEGORIES = FIGHT_CATEGORIES | {'lane_trade', 'skill_dodge', 'positioning', 'escape', 'macro_decision'}
# Kept as a compatibility alias for older exploratory imports.
POTENTIAL = COACH_CATEGORIES
COACHING_SIGNALS = {
    'skill_dodge', 'skill_hit', 'positioning_good', 'positioning_risk',
    'trade_good', 'trade_bad', 'escape', 'gank_response',
    'kill_participation', 'wave_decision', 'recall_timing',
}
PLAYER_RELEVANCE = {'direct', 'indirect', 'unclear', 'none'}


def _is_eligible(row: dict, minimum_score: int = 5) -> bool:
    if row.get('score', 0) < minimum_score or row.get('player_relevance') == 'none':
        return False
    if row.get('category') in COACH_CATEGORIES:
        return True
    # Non-fight phases (including fountain/recall) can be a decision to coach,
    # but not on category alone: require visual evidence and a concrete signal.
    return (row.get('importance', 0) >= 2 and
            bool(row.get('coaching_signals')) and bool(row.get('evidence_images')))


def _candidate_score(kind: str, importance: int, signals: list[str],
                     image_refs: list[str], relevance: str) -> int:
    if kind == 'unknown' or relevance == 'none':
        return 0
    evidence_bonus = 2 * min(2, len(signals)) if image_refs else 0
    # Idle farming, walking, death screen and recall are not automatically bad;
    # but without an observed decision there is nothing yet to prioritize.
    if kind not in COACH_CATEGORIES and not (importance >= 2 and signals and image_refs):
        return 0
    return BASE_SCORE[kind] + importance + evidence_bonus

def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def build_windows(duration: float, window_sec: float=24., step_sec: float=4.):
    """Cover the full VOD, including its final partial window; no first-N break."""
    if not 12 <= window_sec <= 60:
        raise ValueError('window_sec must be in [12, 60] seconds')
    if not 1 <= step_sec <= window_sec/2:
        raise ValueError('step_sec must be >=1 and <=window_sec/2')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid video duration')
    out = []
    for n in range(int(math.ceil(duration/window_sec))):
        start = n * window_sec
        end = min(duration, (n + 1) * window_sec)
        points = []
        idx = 0
        t = start + min(step_sec/2, (end-start)/2)
        while t < end - 1e-6:
            points.append({'image_id':f'w{n+1:04d}_f{idx+1:02d}', 'sec':round(t,3), 'timecode':timecode(t)})
            idx += 1
            t += step_sec
        if not points:
            points = [{'image_id':f'w{n+1:04d}_f01','sec':round((start+end)/2,3),
                       'timecode':timecode((start+end)/2)}]
        out.append({'window_id':f'w{n+1:04d}', 'start_sec':round(start,3),
                    'end_sec':round(end,3), 'images':points})
    return out


def _extract_frames(video:Path, video_info:dict, windows:list, folder:Path, max_edge:int=960):
    """Save downscaled images for reproducible storyboard inputs."""
    import cv2
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError(f'Cannot decode {video}')
    folder.mkdir(parents=True,exist_ok=True)
    all_images=sorted((im for w in windows for im in w['images']),key=lambda x:x['sec'])
    try:
        for image in all_images:
            name=image['image_id']+'.jpg'; path=folder/name
            if path.is_file() and path.stat().st_size>2000:
                image['file']=str(path.relative_to(ROOT));image['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                continue
            index=min(video_info['frame_count']-1,max(0,round(image['sec']*video_info['fps'])))
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Could not read frame index {index} at {image["timecode"]}')
            h,w=frame.shape[:2]
            scale=min(1.,max_edge/max(w,h))
            if scale<1:frame=cv2.resize(frame,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(path),frame,[int(cv2.IMWRITE_JPEG_QUALITY),77]):
                raise RuntimeError(f'Could not save {path}')
            image['file']=str(path.relative_to(ROOT))
            image['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    finally:cap.release()


def _response_json(body:str):
    try:return json.loads(body)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', body.strip(),flags=re.I))


def _gemini_json(model:str,prompt:str,windows:list,retries:int=2):
    from google import genai
    from google.genai import types
    key=os.getenv('GEMINI_API_KEY')
    if not key:raise EnvironmentError('Missing GEMINI_API_KEY. Set it in project .env')
    client=genai.Client(api_key=key,http_options=types.HttpOptions(timeout=180000))
    parts=[types.Part.from_text(text=prompt)]
    for w in windows:
        parts.append(types.Part.from_text(text=f"WINDOW {w['window_id']} | MP4 {timecode(w['start_sec'])} - {timecode(w['end_sec'])}"))
        for im in w['images']:
            p=ROOT/im['file']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=im['sha256']:
                raise RuntimeError(f'Storyboard image changed: {p}')
            parts.append(types.Part.from_text(text=f"IMAGE {im['image_id']} MP4 {im['timecode']}"))
            parts.append(types.Part.from_bytes(data=p.read_bytes(),mime_type='image/jpeg'))
    for attempt in range(retries+1):
        try:
            response=client.models.generate_content(
                model=model,contents=[types.Content(role='user',parts=parts)],
                config=types.GenerateContentConfig(temperature=0,response_mime_type='application/json'))
            if not response.text:raise RuntimeError('Empty Gemini response')
            obj=_response_json(response.text)
            if not isinstance(obj,dict) or not isinstance(obj.get('windows'),list):
                raise RuntimeError('Gemini response must be JSON object with windows list')
            return obj
        except Exception as e:
            code=str(getattr(e,'code',''))
            status=str(getattr(e,'status',''))
            message=str(e)
            retryable= bool(re.search(r'\b(408|429|500|502|503|504)\b|UNAVAILABLE|RESOURCE_EXHAUSTED',
                                      ' '.join((code,status,message)),re.I))
            if attempt==retries or not retryable:raise
            delay=min(20,2**(attempt+1));print(f'[RETRY] Gemini {str(e)[:100]} ({delay}s)',flush=True)
            time.sleep(delay)
    raise RuntimeError('Unreachable')


def _scout_prompt(windows:list,champion:str,role:str):
    ids=[w['window_id'] for w in windows]
    return f'''당신은 LoL VOD에서 **플레이어에게 코칭할 가치가 있는 장면의 후보**를 찾는 분석기입니다.
각 WINDOW는 24초 안팎이고 이미지는 약 4초 간격입니다. 이미지 사이의 스킬과 움직임은 관찰되지 않았을 수 있습니다.
시점 챔피언: {champion}; 역할: {role}. 실제 이미지가 다르면 이미지를 우선하세요.

대상은 킬/교전뿐 아니라 스킬 회피·피격, 위험하거나 좋은 포지셔닝, 딜교환, 생존/도주,
갱킹 대응, 라인 관리와 귀환 판단입니다. 킬이 없다는 이유로 낮은 중요도를 매기지 마세요.
다만 평범한 파밍, 근거 없는 이동, 단순 우물 대기에는 높은 중요도를 주지 마세요.
우물·귀환 화면 하나만 보고 구간 전체를 낮게 평가하지 마세요. 앞뒤 이미지의 맥락을 함께 보세요.
아군 사망 알림만으로 시점 플레이어가 교전했다고 주장하지 마세요.
킬 주체, Q/E 시전, 스킬 적중·회피, CC 종류, 인과관계는 연속 프레임에 근거가 없으면 단정하지 마세요.
불명확하지만 코칭할 여지가 있으면 'unclear' 및 why에 추가 확인 필요를 적으세요.

각 WINDOW마다 다음 필드를 **모두** 반환하세요.
category: {', '.join(sorted(CATEGORIES))}
importance: 0=거의 코칭 가치 없음, 1=작은 참고, 2=코칭 가능, 3=분명한 판단 개선/모범 사례,
4=핵심 생존·포지셔닝·교전 사례, 5=경기에 큰 영향을 주는 결정적 장면.
player_relevance: direct=시점 플레이어 본인 행동, indirect=플레이어 판단과 연결된 주변 상황,
unclear=확신 불가, none=플레이어와 무관한 다른 사람 사건.
coaching_signals: 실제 이미지 근거로 가능한 태그만 사용: {', '.join(sorted(COACHING_SIGNALS))}
evidence_images: 제공된 IMAGE ID 중 근거 이미지. 없다면 빈 배열.
why: 관찰 사실과 추정을 분리하여 한국어 한 문장으로 작성.

JSON 형식: {{"windows":[{{"window_id":"w0001","category":"positioning","importance":3,
"player_relevance":"direct","coaching_signals":["positioning_risk"],
"evidence_images":["w0001_f02"],"why":"플레이어가 적의 견제 범위에 오래 머무는 것으로 보여 추가 확인 가치가 있음"}}]}}
모든 WINDOW IDs 각각 하나씩 반환: {json.dumps(ids,ensure_ascii=False)}
'''


def _verify_prompt(windows:list,champion:str,role:str):
    return f'''LoL 코칭 후보 **2차 독립 검증**입니다. 앞 단계의 분류는 제공하지 않습니다.
시점 챔피언: {champion}; 역할: {role}. 2초 간격 프레임만 사용합니다.
교전·킬뿐 아니라 스킬 회피/피격, 포지셔닝, 딜교환, 생존, 운영 결정의 코칭 가치를 평가하세요.
우물이나 귀환은 자동으로 저평가하지 말고 전후 맥락을 보세요.
보이지 않는 순간의 스킬 시전·스킬 회피·킬 주체·갱킹 예측을 단정하지 마세요.
근거 없으면 중요도를 낮추고 불확실성을 명시하세요. 아군의 교전과 본인 참여를 구분하세요.
category: {', '.join(sorted(CATEGORIES))}
importance=0..5 (교전 여부가 아니라 코칭 가치).
player_relevance=direct|indirect|unclear|none.
coaching_signals는 다음에서만 고르세요: {', '.join(sorted(COACHING_SIGNALS))}.
evidence_images는 실제 제공된 IMAGE ID만 쓰세요.
why에는 증거와 미확인 사항을 함께 요약하세요.
JSON: {{"windows":[{{"window_id":"w0001","category":"lane_trade","importance":2,
"player_relevance":"direct","coaching_signals":["trade_bad"],
"evidence_images":["w0001_d03"],"why":"공격을 받는 장면이 확인되나 정확한 스킬 종류는 미확인"}}]}}
모든 WINDOW를 반환하세요: {json.dumps([w['window_id'] for w in windows],ensure_ascii=False)}
''' 

def normalize_response(obj:dict, windows:list):
    """Normalize untrusted model JSON without upgrading missing evidence to a fact."""
    expected={w['window_id']:w for w in windows}
    by_id={}
    for row in obj.get('windows',[]):
        if not isinstance(row,dict):continue
        wid=row.get('window_id')
        if wid not in expected or wid in by_id:continue
        kind=row.get('category','unknown')
        if kind not in CATEGORIES:kind='unknown'
        try:importance=int(row.get('importance',0))
        except (ValueError,TypeError):importance=0
        importance=max(0,min(5,importance))
        image_ids={im['image_id'] for im in expected[wid]['images']}
        refs=row.get('evidence_images',[])
        if not isinstance(refs,list):refs=[]
        image_refs=list(dict.fromkeys(x for x in refs if isinstance(x,str) and x in image_ids))[:8]
        raw_signals=row.get('coaching_signals',[])
        if not isinstance(raw_signals,list):raw_signals=[]
        signals=list(dict.fromkeys(x for x in raw_signals if isinstance(x,str) and x in COACHING_SIGNALS))
        relevance=row.get('player_relevance','unclear')
        if relevance not in PLAYER_RELEVANCE:relevance='unclear'
        by_id[wid]={'window_id':wid, 'category':kind, 'importance':importance,
                    'player_relevance':relevance, 'coaching_signals':signals,
                    'evidence_images':image_refs, 'why':str(row.get('why',''))[:250],
                    'score':_candidate_score(kind,importance,signals,image_refs,relevance)}
    for wid in expected:
        if wid not in by_id:
            by_id[wid]={'window_id':wid,'category':'unknown','importance':0,
                        'player_relevance':'unclear','coaching_signals':[],
                        'evidence_images':[],'why':'No valid model classification returned',
                        'score':0,'missing_response':True}
    return [by_id[w['window_id']] for w in windows]

def _batch_groups(rows:list,n:int):
    return [rows[i:i+n] for i in range(0,len(rows),n)]


def scout_candidates(rows:list, minimum_score:int=5):
    return [r for r in rows if _is_eligible(r,minimum_score)]


def stratified_shortlist(rows:list,window_map:dict,limit:int=10,duration:float=1.):
    """Global coverage with reserved verification capacity for non-fight coaching."""
    ordered=sorted((r for r in rows if _is_eligible(r)),
                   key=lambda r:(-r['score'],-r['importance'],window_map[r['window_id']]['start_sec']))
    chosen=[];seen=set()
    def add(r):
        if r['window_id'] not in seen and len(chosen)<limit:
            chosen.append(r);seen.add(r['window_id'])
    # Reserve limited dense-verify slots for good non-fight coaching candidates.
    # No quota is filled if no qualifying candidate exists.
    reserve=min(2, limit//4)
    nonfight=[r for r in ordered if r['category'] not in FIGHT_CATEGORIES]
    for r in nonfight[:reserve]:add(r)
    for bucket in range(6):
        matching=[r for r in ordered if
                  min(5,int(6*window_map[r['window_id']]['start_sec']/max(duration,1)))==bucket]
        if matching:add(matching[0])
    for r in ordered:add(r)
    return chosen

def _dense_windows(windows:list, short:list,video_info:dict, dest:Path,dense_step:float=2.):
    src={w['window_id']:w for w in windows}
    denser=[]
    for row in short:
        w=src[row['window_id']]
        t=w['start_sec']+min(dense_step/2,(w['end_sec']-w['start_sec'])/2)
        points=[];index=0
        while t < w['end_sec']-1e-6:
            points.append({'image_id':f"{w['window_id']}_d{index+1:02d}",
                           'sec':round(t,3),'timecode':timecode(t)})
            index+=1;t+=dense_step
        denser.append({**{k:v for k,v in w.items() if k!='images'},'images':points})
    return denser


def _rank_verified(windows:list, rows:list, verified:dict, top_k:int=5, before:float=5.,after:float=8.):
    """Merge adjacent verified coaching windows; rank without a fight-only gate."""
    winmap={w['window_id']:w for w in windows}
    candidates=[]
    for sc in rows:
        vid=sc['window_id'];vr=verified.get(vid)
        if not vr or not _is_eligible(vr):continue
        w=winmap[vid]
        score=vr['score']*2+sc['score']
        candidates.append({**vr,'scout_category':sc['category'],'scout_score':sc['score'],
                           'combined_score':score,'window_start_sec':w['start_sec'],
                           'window_end_sec':w['end_sec']})
    chron=sorted(candidates,key=lambda r:r['window_start_sec'])
    grouped=[]
    for r in chron:
        if grouped and r['window_start_sec']<=grouped[-1]['end']+0.5 and (r['window_end_sec']-grouped[-1]['start'])<=80:
            grouped[-1]['members'].append(r)
            grouped[-1]['end']=max(grouped[-1]['end'],r['window_end_sec'])
        else:
            grouped.append({'start':r['window_start_sec'],'end':r['window_end_sec'],'members':[r]})
    ranked=sorted(grouped,key=lambda g:(-max(r['combined_score'] for r in g['members']) - min(2,len(g['members'])-1),g['start']))
    # When there is independently verified, consequential non-combat coaching,
    # allow at least one such scene in the Top-K. No artificial quota if absent.
    qualifying_nonfight=[g for g in ranked if
        all(r['category'] not in FIGHT_CATEGORIES for r in g['members']) and
        any(r['importance']>=3 and r['evidence_images'] and r['coaching_signals']
            for r in g['members'])]
    preferred=qualifying_nonfight[:1]+[g for g in ranked if g not in qualifying_nonfight[:1]]
    selected=[]
    duration=windows[-1]['end_sec']
    for g in preferred:
        if len(selected)>=top_k:break
        st=max(0,g['start']-before)
        en=min(duration,g['end']+after)
        if any(min(en,s['end_sec'])-max(st,s['start_sec'])>1 for s in selected):continue
        primary=max(g['members'],key=lambda r:r['combined_score'])
        selected.append({'scene_id':f'scene_{len(selected)+1:03d}',
                         'start_sec':round(st,3),'end_sec':round(en,3),
                         'origin':'vlm_coaching_storyboard_and_dense_candidate_UNVERIFIED',
                         'verified_combat':False,
                         'window_id':primary['window_id'],
                         'source_window_ids':[r['window_id'] for r in g['members']],
                         'detector_score_not_accuracy':primary['combined_score'],
                         'detector_category':primary['category'],
                         'player_relevance':primary['player_relevance'],
                         'coaching_signals':primary['coaching_signals'],
                         'why':primary['why']})
    return selected, sorted(candidates,key=lambda r:(-r['combined_score'],r['window_start_sec']))

def _save_csv(path:Path,rows:list, cols:list):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        wr.writeheader()
        for r in rows:wr.writerow(r)


def _create_contact_sheet(video:Path,info:dict,scenes:list,out:Path):
    if not scenes:return
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    width=4*300; h=200
    canvas=Image.new('RGB',(width, len(scenes)*h),(23,28,32))
    draw=ImageDraw.Draw(canvas)
    cap=cv2.VideoCapture(str(video))
    try:
        for i,scene in enumerate(scenes):
            top=i*h
            for j,at in enumerate([scene['start_sec'],
                    scene['start_sec']+(scene['end_sec']-scene['start_sec'])/3,
                    scene['start_sec']+2*(scene['end_sec']-scene['start_sec'])/3,
                    scene['end_sec']]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(info['frame_count']-1,round(at*info['fps'])))
                ok,frame=cap.read()
                if not ok:continue
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                hh,ww=frame.shape[:2];scale=min(300/ww,160/hh)
                im=Image.fromarray(frame).resize((round(ww*scale),round(hh*scale)))
                canvas.paste(im,(j*300,top+20))
                draw.text((j*300+3,top+180),timecode(at),fill=(255,255,255))
            draw.text((8,top+3),
              f"{scene['scene_id']} | {scene['window_id']} | score {scene['detector_score_not_accuracy']} | {scene['detector_category']}",
              fill=(242,213,113))
    finally:cap.release()
    out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out,quality=88)


def discover(video:Path, discovery_id:str, model:str='gemini-3.6-flash',
             champion:str='unknown',role:str='unknown',execute:bool=False,
             window_sec:float=24.,step_sec:float=4.,dense_step:float=2.,
             batch_windows:int=4,verify_top:int=10,top_k:int=5,max_requests:int=40):
    discovery_id=safe_id(discovery_id)
    video=video.resolve()
    if not video.is_file():raise FileNotFoundError(video)
    if not 1<=batch_windows<=6:raise ValueError('batch_windows: 1..6')
    if not 1<=verify_top<=24:raise ValueError('verify_top: 1..24')
    if not 1<=top_k<=15:raise ValueError('top_k: 1..15')
    if not 1<=dense_step<=4:raise ValueError('dense_step: 1..4')
    if max_requests < 1:raise ValueError('max_requests must be positive')
    meta=video_meta(video)
    windows=build_windows(meta['duration_sec'],window_sec,step_sec)
    calls_scout=math.ceil(len(windows)/batch_windows)
    calls_dense=math.ceil(verify_top/2)
    # Estimate before extracting frames, so no unexpected cost or long process.
    if execute and calls_scout+calls_dense>max_requests:
        raise ValueError(f'Planned worst-case {calls_scout+calls_dense} API calls exceeds --max-requests {max_requests}. '
                          'Adjust window_sec / batch_windows / verify_top, or explicitly raise limit. '
                          'Transient retries may add extra calls.')
    out=ROOT/'results'/'discovery'/discovery_id
    spec={'schema':'loldov-discovery-v4','prompt_version':PROMPT_VERSION,
          'video_path':str(video),'video_fingerprint_partial_sha256':file_fingerprint(video),
          'video_meta':meta,'model':model,'champion':champion,'role':role,
          'window_sec':window_sec,'step_sec':step_sec,'dense_step':dense_step,
          'batch_windows':batch_windows,'verify_top':verify_top,'top_k':top_k,
          'scout_batch_count':calls_scout,'dense_batch_count_at_most':calls_dense}
    spec_path=out/'scan_spec.json'
    if spec_path.is_file():
        existing=load(spec_path)
        if _sha_obj({k:v for k,v in existing.items() if k!='created_at'})!=_sha_obj(spec):
            raise RuntimeError(f'Existing discovery ID {discovery_id} uses different video/config/model. '
                               'Use a fresh --discovery-id. No files were overwritten.')
    else:
        out.mkdir(parents=True,exist_ok=True)
        dump(spec_path,{**spec,'created_at':_utc()})
    _extract_frames(video,meta,windows,out/'storyboard',max_edge=960)
    dump(out/'storyboard_index.json',{'windows':windows,
         'note':'Every window of the video was sampled; brief events between frames can still be missed.'})
    (out/'SCANNING_NOTICE.txt').write_text(
        'This is experimental VLM candidate discovery, NOT verified combat ground truth.\n'
        'Do NOT report scores as classification accuracy.\n'
        'All evidence images are from the video only; no manually labeled ground truth was used.\n',encoding='utf-8')
    if not execute:
        dump(out/'scan_plan.json',{'status':'ready_no_api_calls','duration_sec':meta['duration_sec'],
             'window_count':len(windows),'storyboard_image_count':sum(len(w['images']) for w in windows),
             'planned_scout_calls':calls_scout,'planned_dense_calls_at_most':calls_dense,
             'planned_api_calls_at_most':calls_scout+calls_dense,
             'instructions':'Add --execute to call Gemini. Costs and retries possible.'})
        return out
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT/'.env')
    except ImportError:pass
    if not os.getenv('GEMINI_API_KEY'):
        raise EnvironmentError(f'Missing GEMINI_API_KEY. Add to {ROOT/".env"} (never commit).')
    # Stage 1: whole video, no early termination.
    scout_batches=_batch_groups(windows,batch_windows)
    scout_rows=[];scout_errors=[]
    for i,batch in enumerate(scout_batches,1):
        p=out/'scout_batches'/f'batch_{i:03d}.json'
        if p.is_file():
            cached=load(p)
            rows=cached['rows']
            if [r['window_id'] for r in rows]!=[w['window_id'] for w in batch]:
                raise RuntimeError('Stale scouting cache: ids differ. Use a NEW discovery id')
        else:
            try:
                print(f'[SCOUT {i}/{len(scout_batches)}] {timecode(batch[0]["start_sec"])} ~ {timecode(batch[-1]["end_sec"])}',flush=True)
                reply=_gemini_json(model,_scout_prompt(batch,champion,role),batch)
                rows=normalize_response(reply,batch)
                if any(r.get('missing_response') for r in rows):
                    raise RuntimeError('Gemini scouting response omitted window IDs; not cached. Rerun to retry.')
                dump(p,{'schema':'scout_batch_v2','windows':[w['window_id'] for w in batch],
                        'rows':rows,'model':model,'created_at':_utc()})
            except Exception as exc:
                error={'stage':'scout','batch':i,'error_type':type(exc).__name__,'message':str(exc)}
                scout_errors.append(error)
                dump(out/'errors'/f'scout_{i:03d}_error.json',error)
                print(f'[ERROR] Scout batch {i}: {str(exc)[:180]}',flush=True)
                continue
        scout_rows.extend(rows)
    _save_csv(out/'all_windows.csv',
              [{**r,'start_sec':w['start_sec'],'end_sec':w['end_sec']} for w in windows
               for r in scout_rows if r['window_id']==w['window_id']],
              ['window_id','start_sec','end_sec','category','importance','player_relevance','coaching_signals','score','why','evidence_images','missing_response'])
    if scout_errors or len(scout_rows)!=len(windows) or any(r.get('missing_response') for r in scout_rows):
        dump(out/'scan_status.json',{'status':'partial_scout','classified_windows':len(scout_rows),
                                    'expected_windows':len(windows),'errors':scout_errors,
                                    'note':'Rerun the SAME command to resume cached batches. Do not prepare.'})
        return out
    shortlist=stratified_shortlist(scout_rows,{w['window_id']:w for w in windows},verify_top,meta['duration_sec'])
    if not shortlist:
        dump(out/'discovery.json',{'schema':'loldov-discovery-result-v4','status':'complete_no_candidates',
                                  'discovery_id':discovery_id,'video_path':str(video),
                                  'video_fingerprint_partial_sha256':file_fingerprint(video),
                                  'windows_scanned':len(windows),'selected_scenes':[],
                                  'warning':'No sufficiently strong VLM candidate; not proof of no coaching-worthy moments.'})
        dump(out/'scan_status.json',{'status':'complete_no_candidates','selected_scenes':0})
        return out
    # Stage 2: denser independent image re-inspection for likely candidate windows.
    dense=_dense_windows(windows,shortlist,meta,out)
    _extract_frames(video,meta,dense,out/'dense_storyboard',max_edge=1100)
    verify_rows=[];verify_errors=[]
    dense_batches=_batch_groups(dense,2)
    for i,batch in enumerate(dense_batches,1):
        p=out/'verify_batches'/f'batch_{i:03d}.json'
        if p.is_file():
            cached=load(p)
            rows=cached['rows']
            if [r['window_id'] for r in rows]!=[w['window_id'] for w in batch]:
                raise RuntimeError('Stale verification cache. Use a new discovery id')
        else:
            try:
                print(f'[VERIFY {i}/{len(dense_batches)}] '+','.join(w['window_id'] for w in batch),flush=True)
                reply=_gemini_json(model,_verify_prompt(batch,champion,role),batch)
                rows=normalize_response(reply,batch)
                if any(r.get('missing_response') for r in rows):
                    raise RuntimeError('Gemini verification response omitted window IDs; not cached. Rerun to retry.')
                dump(p,{'schema':'verify_batch_v2','windows':[w['window_id'] for w in batch],
                        'rows':rows,'model':model,'created_at':_utc()})
            except Exception as exc:
                error={'stage':'verify','batch':i,'error_type':type(exc).__name__,'message':str(exc)}
                verify_errors.append(error)
                dump(out/'errors'/f'verify_{i:03d}_error.json',error)
                print(f'[ERROR] Verify batch {i}: {str(exc)[:180]}',flush=True)
                continue
        verify_rows.extend(rows)
    _save_csv(out/'verified_windows.csv',verify_rows,
              ['window_id','category','importance','player_relevance','coaching_signals','score','why','evidence_images','missing_response'])
    if verify_errors or len(verify_rows)!=len(dense) or any(r.get('missing_response') for r in verify_rows):
        dump(out/'scan_status.json',{'status':'partial_verify','verified_windows':len(verify_rows),
                                    'expected_windows':len(dense),'errors':verify_errors,
                                    'note':'Rerun SAME command to resume. Do not prepare.'})
        return out
    selected,all_candidates=_rank_verified(windows,scout_rows,
                             {r['window_id']:r for r in verify_rows},top_k)
    dump(out/'discovery.json',{'schema':'loldov-discovery-result-v4',
         'status':'complete' if selected else 'complete_no_candidates',
         'discovery_id':discovery_id,'video_path':str(video),
         'video_fingerprint_partial_sha256':file_fingerprint(video),
         'model':model,'model_prompt_version':PROMPT_VERSION,
         'windows_scanned':len(windows),'storyboard_step_sec':step_sec,
         'dense_step_sec':dense_step,'shortlisted_count':len(shortlist),
         'verified_windows_count':len(verify_rows),'selected_scenes':selected,
         'warning':'VLM-selected UNVERIFIED coaching candidates, not factual labels; key moments between snapshots may be missed. No human labels or previous reports used.'})
    _save_csv(out/'all_candidates.csv',all_candidates,
              ['window_id','window_start_sec','window_end_sec','category','importance','scout_category','scout_score',
               'combined_score','why','evidence_images'])
    _save_csv(out/'selected_scenes.csv', selected,
              ['scene_id','start_sec','end_sec','window_id','detector_category','player_relevance',
                'coaching_signals','detector_score_not_accuracy','why','origin'])
    # Independent human evaluation templates. NEVER read by the detector or coaching model.
    _save_csv(out/'human_selection_review_template.csv',
              [{'scene_id':sc['scene_id'],'start_sec':sc['start_sec'],'end_sec':sc['end_sec'],
                'human_actual_champion_fight_0_or_1':'','human_importance_0_to_5':'',
                'human_notes':''} for sc in selected],
              ['scene_id','start_sec','end_sec','human_actual_champion_fight_0_or_1',
               'human_importance_0_to_5','human_notes'])
    _save_csv(out/'human_missed_fights_template.csv',[],
              ['manual_start_sec','manual_end_sec','why_important','detected_by_v4_0_or_1'])
    _create_contact_sheet(video,meta,selected,out/'selected_contact_sheet.jpg')
    dump(out/'scan_status.json',{'status':'complete' if selected else 'complete_no_candidates',
                                 'windows_scanned':len(windows),'selected_scenes':len(selected),
                                 'selected_contact_sheet':str(out/'selected_contact_sheet.jpg'),
                                 'note':'Review selected scenes against real video before A/B; unverified candidates.'})
    return out


def load_discovered_scenes(discovery_id:str,video:Path):
    data=load(ROOT/'results'/'discovery'/safe_id(discovery_id)/'discovery.json')
    if data.get('status')!='complete':
        raise RuntimeError('Discovery incomplete or no valid candidates. Review scan_status.json')
    if Path(data['video_path']).resolve()!=video.resolve() or file_fingerprint(video)!=data['video_fingerprint_partial_sha256']:
        raise RuntimeError('Discovery used a different video. Refusing scene reuse.')
    scenes=[]
    for row in data['selected_scenes']:
        scenes.append({'scene_id':safe_id(row['scene_id']),
                       'start_sec':float(row['start_sec']),'end_sec':float(row['end_sec']),
                       'origin':'vlm_discovery_candidate_UNVERIFIED',
                       'verified_combat':False,
                       'discovery_id':safe_id(discovery_id),
                       'source_window_id':row['window_id']})
    if not scenes:raise RuntimeError('No scenes to prepare. No proof there are no fights.')
    return scenes
