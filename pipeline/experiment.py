"""Gemini image-only A/B experiment. Human labels never go into either request."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from .core import ROOT,load,dump,safe_id,file_fingerprint,augment_focus_frames
from .evidence import compile_evidence

PROMPT_VERSION='loldov_coaching_ab_v1'
COMMON_RULES = '''당신은 리그 오브 레전드 소환사의 협곡 VOD 코치입니다.
아래에 제공하는 모든 시각은 녹화 MP4 재생 시각이며 게임 HUD 시각과 혼동하지 마세요.
프레임은 전체 영상이 아니라 간격을 두고 추출한 샘플입니다.
프레임만으로 관찰 가능한 상황과 추정·미확인을 엄격하게 분리하세요.
영웅의 사망 여부, CC 종류, 사용된 스킬의 정확한 순서, 킬 관여, 적 스킬 쿨타임,
시야/와드, D/F 주문 이름, HP 숫자 등을 억지로 단정하지 마세요.
불확실하다면 unknown 또는 확인 불가라고 쓰되, 해당 장면에서 실천할 수 있는
조건부 코칭을 최대 3개까지 구체적으로 제안하세요.
각 코칭은 실제 제공된 frame_id와 시각을 근거로 명시해야 합니다.
입력에 있는 후보 분석이나 게임 일반 규칙도 실제 영상 속 사건을 자동으로 증명하지 않습니다.
응답은 반드시 아래 구조를 갖는 JSON 객체로 작성하세요(스키마에 없는 추가 필드 금지 안 함):
{
 "summary": "장면 요약",
 "scenes": [{"scene_id":"...","observations":[{"frame_id":"frame_000.jpg","timecode":"00:00:00.000","description":"화면에서 관찰된 행동", "certainty":"visible|uncertain"}], "unknowns":["미확인 사실"]}],
 "coaching": [{"scene_id":"...", "advice":"행동 가능한 개선 방향", "evidence_frames":["scene_001/frame_000.jpg"], "reason":"근거와 조건", "unknowns":["아직 모르는 정보"]}],
 "limitations":["해석 제한"]
}
'''


def common_header(manifest:dict):
    scenes=[]
    for sc in manifest['scenes']:
        scenes.append({'scene_id':sc['scene_id'],'window_mp4_sec':[sc['start_sec'],sc['end_sec']],
            'sampled_frames':[{'frame_ref':f"{sc['scene_id']}/{fr['frame_id']}",
            'timecode_mp4':fr['source_timecode']} for fr in sc['frames']]})
    return ('분석 대상: '+json.dumps({'video_id':manifest['video_id'],
              'champion_user_declared':manifest.get('champion','unknown'),
              'role_user_declared':manifest.get('role','unknown'),
              'scenes':scenes},ensure_ascii=False)+'\n')


def prompts(manifest:dict,evidence:dict):
    common = COMMON_RULES+'\n'+common_header(manifest)
    baseline=common+'\n[입력 조건: 원본 이미지 프레임만 사용. 외부 HUD/OCR/RAG 분석 내용 없음.]\n'
    auxiliary=json.dumps(evidence,ensure_ascii=False,indent=2)
    assisted=(common+'\n[입력 조건: 동일한 원본 이미지 + 자동 보조 분석 후보 + 일반 게임 규칙]\n'
       '아래 정보는 **정답이 아니라 자동 검출 후보**입니다. 후보가 없는 것은 사건의 부재가 아니며, '
       '원본 이미지와 충돌할 경우 이미지를 우선하고, 확인되지 않은 숫자/스킬 시전/CC를 만들어내지 마세요.\n'
       '<auxiliary_machine_evidence>\n'+auxiliary+'\n</auxiliary_machine_evidence>\n')
    return {'baseline':baseline,'assisted':assisted}


def ordered_frames(manifest):
    frames=[]
    for sc in manifest['scenes']:
        for fr in sc['frames']:
            p=ROOT/fr['frame_file']
            if not p.is_file():raise FileNotFoundError(p)
            if hashlib.sha256(p.read_bytes()).hexdigest()!=fr['sha256']:
                raise RuntimeError(f'Image changed after prepare: {p}')
            frames.append((sc['scene_id'],fr,p))
    return frames


def _response_json(text:str):
    try:return json.loads(text)
    except json.JSONDecodeError:
        cleaned=re.sub(r'^```(?:json)?\s*|\s*```$', '',text.strip(),flags=re.I)
        return json.loads(cleaned)


def gemini_request(model:str,prompt:str,frames:list,retries=2):
    from google import genai
    from google.genai import types
    key=os.getenv('GEMINI_API_KEY')
    if not key:raise EnvironmentError('Missing GEMINI_API_KEY in project .env')
    client=genai.Client(api_key=key,http_options=types.HttpOptions(timeout=120000))
    parts=[types.Part.from_text(text=prompt)]
    for scene_id,fr,path in frames:
        parts.append(types.Part.from_text(text=f"FRAME {scene_id}/{fr['frame_id']} | MP4 {fr['source_timecode']}"))
        parts.append(types.Part.from_bytes(data=path.read_bytes(),mime_type='image/jpeg'))
    for attempt in range(retries+1):
        try:
            response=client.models.generate_content(
                model=model,contents=[types.Content(role='user',parts=parts)],
                config=types.GenerateContentConfig(temperature=0.0,response_mime_type='application/json'))
            if not response.text:
                raise RuntimeError('Gemini returned empty response text')
            obj=_response_json(response.text)
            if not isinstance(obj,dict):raise RuntimeError('Gemini response must be a JSON object')
            return {'raw_text':response.text,'parsed':obj,
                    'model_used':model,'attempts':attempt+1}
        except Exception as exc:
            # Only retry transient overload/rate limit; do not retry bad auth/config.
            code=getattr(exc,'code',None)
            status=str(getattr(exc,'status',''))
            desc=str(exc)
            retryable=(str(code) in {'408','429','500','502','503','504'} or
                       bool(re.search(r'\b(429|503|504|502|500|408)\b|UNAVAILABLE|RESOURCE_EXHAUSTED',status+' '+desc,re.I)))
            if attempt>=retries or not retryable:
                raise
            delay=min(12,2**(attempt+1));print(f'[RETRY] Gemini transient error, delay {delay}s: {str(exc)[:160]}',flush=True)
            time.sleep(delay)
    raise RuntimeError('unreachable')


def _write_review(run_dir:Path,manifest:dict,outputs:dict):
    # Both arms use exactly the same frame_ids; no external annotations as input.
    path=run_dir/'comparison_template.csv'
    cols=['criterion','baseline_score_0_2_or_NA','assisted_score_0_2_or_NA','evidence_and_notes']
    criteria=[
       ('실제 프레임과 사실 일치','실제 영상에서 관찰 가능한 주장인가'),
       ('수치·스킬·CC 추론 절제','근거 불충분한 숫자/상태 단정을 피하는가'),
       ('중요 장면 누락','중요한 교전/CC 후보를 놓치지 않는가'),
       ('환각 또는 과도한 추론','보이지 않는 스킬 시전·킬 관여 등을 만들어내지 않는가'),
       ('코칭 구체성','지정한 장면과 프레임 근거를 연결하는가'),
       ('코칭 실천 가능성','조건부로 실천 가능한 조언인가'),
       ('근거 추적','제공된 frame_id/MP4 시각을 확인할 수 있는가'),
       ('불확실성 구분','visible/uncertain/unknown의 구분이 적절한가')]
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(cols)
        for name,desc in criteria:writer.writerow([f'{name} — {desc}','','',''])
    md=['# 동일 영상 A/B 코칭 결과 비교','',
       f"- Video ID: `{manifest['video_id']}`",
       f"- Prompt version: `{PROMPT_VERSION}`",
       '- **동일 프레임, 동일 공통 프롬프트** / assisted만 자동 후보·일반 지식 추가',
       '- 이 문서는 결과 비교 준비물일 뿐 자동 정확도 점수/향상률을 주장하지 않습니다.',
       '- 평가를 위해 사람 정답을 별도로 사용하되 프롬프트로 입력하지 마세요.',
       '- 기존 사람이 교정한 리포트와 비교하면 자동 모델 성능 실험이 아닙니다.',
       '', '## 입력 조건', '',
       '| Arm | 입력 | API 결과 |', '|---|---|---|',
       f"| baseline | 이미지 | {outputs.get('baseline','pending')} |",
       f"| assisted | 동일 이미지 + 자동 근거 + 일반 규칙 | {outputs.get('assisted','pending')} |",'',
       '## 자동으로 판정할 수 없는 것','',
       '실제 코칭 정확도, 사람 정답과의 일치율, 장면 전체 CC 재현율, 다른 챔피언 일반화 성능.','',
       '## 점수표', '', '같은 폴더의 `comparison_template.csv`에서 영상 근거와 비교하여 채점하세요.']
    (run_dir/'comparison_guide.md').write_text('\n'.join(md),encoding='utf-8')


def _validate_refs(result:dict,frames):
    valid={f'{sc}/{fr["frame_id"]}' for sc,fr,_ in frames}
    problems=[]
    for advice in result.get('coaching',[]):
        if not isinstance(advice,dict):continue
        for ref in advice.get('evidence_frames',[]):
            if ref not in valid:problems.append(ref)
    return {'unrecognized_frame_references':problems,
            'warning':'Mechanical reference validation only; does not check truth of observed claims.'}


def run(video_id:str,model:str,run_tag:str,execute:bool=False,force=False):
    video_id=safe_id(video_id);run_tag=safe_id(run_tag)
    if force:
        raise ValueError('--force is disabled in refactor v2: preserve previous model output and use a new --run-tag')
    base=ROOT/'results'/'pipeline'/video_id
    manifest=load(base/'manifest.json')
    if file_fingerprint(Path(manifest['video_path']))!=manifest['video_fingerprint_partial_sha256']:
        raise RuntimeError('Video has changed; prepare with a new video_id after review')
    manifest=augment_focus_frames(manifest)
    ev=compile_evidence(manifest)
    frame_rows=ordered_frames(manifest)
    text=prompts(manifest,ev)
    out=base/'experiments'/run_tag;out.mkdir(parents=True,exist_ok=True)
    prompt_hashes={arm:hashlib.sha256(prompt.encode('utf-8')).hexdigest() for arm,prompt in text.items()}
    snapshot_hash=hashlib.sha256(json.dumps(ev,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()
    old_audit_path=out/'experiment_manifest.json'
    if old_audit_path.is_file() and any((out/f'{arm}_result.json').is_file() for arm in ('baseline','assisted')) and not force:
        old=load(old_audit_path)
        if (old.get('prompt_hashes')!=prompt_hashes or old.get('model')!=model or
            old.get('evidence_snapshot_sha256')!=snapshot_hash):
            raise RuntimeError('Existing model result uses different prompt/model/evidence. Use a NEW --run-tag (e.g. ab02) instead of silently mixing runs.')
    for arm,prompt in text.items():
        (out/f'{arm}_prompt.txt').write_text(prompt,encoding='utf-8')
    dump(out/'auxiliary_evidence_snapshot.json',ev)
    audit={'schema':'loldov-experiment-v1','video_id':video_id,'model':model,
           'prompt_version':PROMPT_VERSION,'run_tag':run_tag,
           'same_frames_for_both_arms':True,
           'frame_reference_hashes':[{'frame':f'{s}/{f["frame_id"]}','sha256':f['sha256']} for s,f,p in frame_rows],
           'baseline_extra_evidence':False,'assisted_extra_evidence':True,
           'manual_labels_injected':False,
           'prompt_hashes':prompt_hashes,'evidence_snapshot_sha256':snapshot_hash,
           'difference_is_auxiliary_bundle_not_isolated_single_HUD_component':True,
           'timestamp_utc':datetime.now(timezone.utc).isoformat()}
    dump(out/'experiment_manifest.json',audit)
    if not execute:
        _write_review(out,manifest,{'baseline':'not_sent_dry_run','assisted':'not_sent_dry_run'})
        return out
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT/'.env')
    except ImportError:
        pass
    if not os.getenv('GEMINI_API_KEY'):
        raise EnvironmentError(f'GEMINI_API_KEY not set. Add to {ROOT / ".env"}. NEVER commit .env.')
    status={}
    for arm in ('baseline','assisted'):
        result_path=out/f'{arm}_result.json'
        if result_path.exists() and not force:
            status[arm]='existing_result_kept';continue
        try:
            print('[GEMINI]',arm,model,flush=True)
            response=gemini_request(model,text[arm],frame_rows)
            response['source_ref_check']=_validate_refs(response['parsed'],frame_rows)
            dump(result_path,response)
            status[arm]='ok'
            error_path=out/f'{arm}_error.json'
            if error_path.exists():error_path.unlink()
        except Exception as exc:
            dump(out/f'{arm}_error.json',{'status':'failed','error_type':type(exc).__name__,
                'message':str(exc),'arm':arm,'retry_note':'Only transient 429/503 errors are retried.'})
            status[arm]='failed';print('[ERROR]',arm,str(exc)[:200],flush=True)
    _write_review(out,manifest,status)
    dump(out/'run_status.json',status)
    return out
