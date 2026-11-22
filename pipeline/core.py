"""Video preparation and scoped, versioned artifacts. No external network/API."""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def dump(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_id(raw: str) -> str:
    name = str(raw).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,90}', name) or '..' in name:
        raise ValueError(f'Unsafe video/scene id: {raw!r}')
    return name


def seconds(raw: str) -> float:
    fields = str(raw).strip().split(':')
    if not 1 <= len(fields) <= 3:
        raise ValueError(f'Invalid time: {raw!r}')
    values = [float(s) for s in fields]
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError(f'Invalid time: {raw!r}')
    if len(values) > 1 and (values[-1] >= 60 or (len(values) == 3 and values[-2] >= 60)):
        raise ValueError(f'Invalid time: {raw!r}')
    value = 0.0
    for x in values:
        value = 60*value+x
    return value


def timecode(t: float) -> str:
    t = max(0, float(t)); h=int(t//3600); m=int((t%3600)//60)
    return f'{h:02d}:{m:02d}:{t%60:06.3f}'


def parse_scene(spec: str, index: int):
    if '=' in spec:
        name, window = spec.split('=', 1)
    else:
        name, window = f'scene_{index:03d}', spec
    name = safe_id(name)
    parts = window.split('-', 1)
    if len(parts)!=2:
        raise ValueError('Scene needs START-END: e.g. 03:08-03:43.467')
    start, end = seconds(parts[0]), seconds(parts[1])
    if start >= end:
        raise ValueError(f'Scene start >= end: {spec!r}')
    return {'scene_id':name,'start_sec':round(start,3),'end_sec':round(end,3),
            'origin':'user_provided_range','verified_combat':False}


def project_path(value: str | Path) -> Path:
    p=Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def video_meta(path: Path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open MP4: {path}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps<=0 or count<=0 or width<=0 or height<=0:
        raise RuntimeError(f'Invalid video info {path}: fps={fps}, frames={count}, size={width}x{height}')
    return {'fps':fps,'frame_count':count,'duration_sec':(count-1)/fps,'width':width,'height':height}


def file_fingerprint(path:Path) -> str:
    """Fast provenance check: file size and first/last MiB; NOT a full cryptographic hash."""
    h=hashlib.sha256()
    size=path.stat().st_size
    with path.open('rb') as f:
        h.update(f.read(1024*1024))
        if size>1024*1024:
            f.seek(max(0,size-1024*1024));h.update(f.read())
    h.update(str(size).encode())
    return h.hexdigest()


def auto_scenes(video:Path, meta:dict, profile:dict, sample_step=.5, drop=.14, pad_before=5.,pad_after=13., max_scenes=5):
    """Weak, HUD health-loss candidates; not a validated fight detector or player recognition."""
    import cv2
    import numpy as np
    cap=cv2.VideoCapture(str(video))
    fps=meta['fps']; samples=[]
    rect=profile['hp_fill_roi_ref']; rw,rh=profile['reference_size']
    if meta['width']/meta['height'] != rw/rh and abs(meta['width']/meta['height']-rw/rh)>.06:
        cap.release();raise ValueError('HUD profile/video aspect ratio mismatches. Supply explicit scenes or a calibrated profile.')
    for sec in np.arange(0,meta['duration_sec']+.001,sample_step):
        n=min(meta['frame_count']-1,int(round(sec*fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES,n); ok,frame=cap.read()
        if not ok:continue
        x0,y0,x1,y1=[round(v*s) for v,s in zip(rect,[meta['width']/rw,meta['height']/rh,meta['width']/rw,meta['height']/rh])]
        crop=frame[y0:y1,x0:x1]
        if crop.size==0:continue
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([34,65,65]),np.array([97,255,255]))>0
        # Column fill fraction and max contiguous-ish fill proportion is a weak proxy.
        col=np.mean(mask,axis=0)>.30
        ratio=float(np.mean(col))
        samples.append((float(sec),ratio))
    cap.release()
    # Compare against prior 2 seconds; only mark potential HP loss, never assert combat.
    candidates=[]
    for i,(t,v) in enumerate(samples):
        previous=[x for tt,x in samples[max(0,i-int(2/sample_step)):i] if t-tt<=2.1]
        if previous and max(previous)-v>=drop and max(previous)>.18:
            if not candidates or t-candidates[-1]>4:
                candidates.append(t)
    windows=[]
    for t in candidates:
        start=max(0,t-pad_before);end=min(meta['duration_sec'],t+pad_after)
        if windows and start<=windows[-1]['end_sec']+2:
            windows[-1]['end_sec']=max(windows[-1]['end_sec'],round(end,3))
            windows[-1]['signal_times_sec'].append(round(t,3))
        else:
            windows.append({'scene_id':f'scene_{len(windows)+1:03d}',
                            'start_sec':round(start,3),'end_sec':round(end,3),
                            'origin':'auto_health_drop_candidate','verified_combat':False,
                            'signal_times_sec':[round(t,3)]})
        if len(windows)>=max_scenes:break
    return windows, {'sampled_points':len(samples),'raw_drop_events':len(candidates),
                     'warning':'HP 하락 기반 후보일 뿐 교전/대상 챔피언/CC를 확정하지 않습니다. 체력 감소 없는 전투는 누락됩니다.'}


def evenly_spaced(start:float,end:float,count:int):
    if count<2:return [start]
    return [start+(end-start)*i/(count-1) for i in range(count)]


def write_scene_frames(cap, meta:dict, scene:dict, dest:Path, max_frames:int):
    import cv2
    dest.mkdir(parents=True,exist_ok=True)
    out=[]; visited=set(); fps=meta['fps']
    for t in evenly_spaced(scene['start_sec'],scene['end_sec'],max_frames):
        frame_number=min(meta['frame_count']-1,max(0,int(round(t*fps))))
        if frame_number in visited:continue
        visited.add(frame_number)
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number)
        ok,frame=cap.read()
        if not ok:continue
        p=dest/f'frame_{len(out):03d}.jpg'
        if not cv2.imwrite(str(p),frame,[int(cv2.IMWRITE_JPEG_QUALITY),91]):
            raise RuntimeError(f'Failed image write: {p}')
        out.append({'frame_file':str(p.resolve().relative_to(ROOT)),
                    'frame_id':p.name,'source_frame_index':frame_number,
                    'source_timestamp_sec':round(frame_number/fps,3),
                    'source_timecode':timecode(frame_number/fps),
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return out


def write_hud_timeline(cap, meta:dict, scene:dict, profile:dict, dest:Path, step:float=.5):
    import cv2
    rw,rh=profile['reference_size'];w,h=meta['width'],meta['height']
    if abs(w/h-rw/rh)>.06:
        raise ValueError('HUD profile aspect ratio incompatible with video')
    rect=profile['ability_row_roi_ref']
    x0,y0,x1,y1=[round(v*s) for v,s in zip(rect,[w/rw,h/rh,w/rw,h/rh])]
    dest.mkdir(parents=True,exist_ok=True)
    frame_rows=[];seen=set(); t=scene['start_sec']
    points=[]
    while t<scene['end_sec']-1e-6:
        points.append(t);t+=step
    points.append(scene['end_sec'])
    for t in points:
        n=min(meta['frame_count']-1,max(0,int(round(t*meta['fps']))))
        if n in seen:continue
        seen.add(n);cap.set(cv2.CAP_PROP_POS_FRAMES,n)
        ok,frame=cap.read()
        if not ok:continue
        roi=frame[y0:y1,x0:x1]
        if roi.size==0:raise RuntimeError(f'Empty HUD ROI {rect}')
        enlarged=cv2.resize(roi,(900,132),interpolation=cv2.INTER_CUBIC)
        image_name=f'hud_{len(frame_rows):03d}_ability_row.png'
        cv2.imwrite(str(dest/image_name),enlarged)
        frame_rows.append({'sample_index':len(frame_rows),'source_frame_index':n,
                           'source_timestamp_sec':round(n/meta['fps'],3),
                           'source_timecode':timecode(n/meta['fps']),
                           'game_timecode':None,'files':{'ability_row':image_name}})
    dump(dest/'hud_timeline.json',{'scene_id':scene['scene_id'],
         'source':'video_frame_sampling', 'hud_profile':profile['profile_id'],
         'frames':frame_rows,'note':'MP4 timecode only. No inferred game clock.'})
    return len(frame_rows)


def prepare(video:Path,video_id:str,champion:str,role:str,scenes:list, profile_path:Path|None,
            max_frames=8, hud_step=.5, do_auto=False, force=False, patch="unknown"):
    import cv2
    video=video.resolve();video_id=safe_id(video_id)
    if not video.is_file():raise FileNotFoundError(video)
    meta=video_meta(video)
    profile=load(profile_path) if profile_path else None
    if profile and abs(meta['width']/meta['height']-profile['reference_size'][0]/profile['reference_size'][1])>.06:
        raise ValueError(f'Profile size mismatch: video {meta["width"]}x{meta["height"]}, profile reference {profile["reference_size"]}')
    if not scenes and not do_auto:raise ValueError('Provide --scene ranges or --auto-scenes')
    auto_info=None
    if do_auto:
        if not profile:raise ValueError('--auto-scenes needs a calibrated --hud-profile')
        scenes,auto_info=auto_scenes(video,meta,profile)
    if not scenes:raise RuntimeError('No candidate scenes found. Use explicit --scene 03:08-03:43.467')
    if len({s['scene_id'] for s in scenes})!=len(scenes):raise ValueError('Scene IDs must be unique')
    for s in scenes:
        if s['end_sec']>meta['duration_sec']+2/meta['fps']:
            raise ValueError(f'Scene beyond video duration: {s}')
    out=ROOT/'results'/'pipeline'/video_id
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'{manifest_path} exists. A video ID must be unique per prepared run. '
                              'Do not overwrite old experiments; use a new --video-id for another preparation.')
    cap=cv2.VideoCapture(str(video)); scene_results=[]
    try:
        for s in scenes:
            frames=write_scene_frames(cap,meta,s,out/'frames'/s['scene_id'],max_frames)
            if not frames:raise RuntimeError(f'No frames extracted for {s["scene_id"]}')
            record={**s,'frames':frames,'hud_samples':0}
            if profile:
                record['hud_samples']=write_hud_timeline(cap,meta,s,profile,out/'hud_temporal'/s['scene_id'],hud_step)
            scene_results.append(record)
    finally:
        cap.release()
    manifest={'schema':'loldov-manifest-v1','created_at':datetime.now(timezone.utc).isoformat(),
        'video_id':video_id,'video_path':str(video),'video_fingerprint_partial_sha256':file_fingerprint(video),
        'video_meta':meta,'champion':champion,'role':role,'patch':str(patch),
        'hud_profile':str(profile_path) if profile_path else None,
        'auto_detector':auto_info,'scenes':scene_results,
        'annotation_policy':'Manual labels, corrections and previous Gemini output NEVER passed to model'}
    dump(manifest_path,manifest)
    return manifest_path


def augment_focus_frames(manifest:dict,max_extra_per_scene:int=3):
    """Add identical auto-CC candidate keyframes to both arms; no human labels read.

    Idempotent. Does not assert CC actually happened; frame selection itself is
    shared preprocessing for both A and B, not a different image input.
    """
    import cv2
    base=ROOT/'results'/'pipeline'/manifest['video_id']
    cap=None;changed=False
    try:
        for scene in manifest['scenes']:
            ctx=base/'hud_temporal'/scene['scene_id']/'hud_cc_context.json'
            if not ctx.is_file():continue
            raw=load(ctx)
            # Only automated candidate timestamp field; never human-verified frames.
            times=[]
            for event in raw.get('candidate_events',[]):
                ts=event.get('timestamp_sec')
                if ts is None and event.get('timecode'):
                    ts=seconds(event['timecode'])
                if ts is None:continue
                ts=float(ts)
                if scene['start_sec'] <= ts <= scene['end_sec']:
                    times.append(ts)
            times=sorted(set(times))[:max_extra_per_scene]
            if not times:continue
            for j,t in enumerate(times,1):
                n=min(manifest['video_meta']['frame_count']-1,
                      int(round(t*manifest['video_meta']['fps'])))
                # Check for the same original frame index, never just near timecode.
                if any(f['source_frame_index']==n for f in scene['frames']):continue
                name=f'frame_focus_{j:03d}.jpg'
                dest=base/'frames'/scene['scene_id']/name
                if cap is None:
                    cap=cv2.VideoCapture(manifest['video_path'])
                    if not cap.isOpened():raise RuntimeError('Cannot reopen video for focus frames')
                cap.set(cv2.CAP_PROP_POS_FRAMES,n)
                ok,frame=cap.read()
                if not ok:continue
                if not cv2.imwrite(str(dest),frame,[int(cv2.IMWRITE_JPEG_QUALITY),91]):
                    raise RuntimeError(f'Cannot save focus frame {dest}')
                fr={'frame_file':str(dest.relative_to(ROOT)),
                    'frame_id':name,'source_frame_index':n,
                    'source_timestamp_sec':round(n/manifest['video_meta']['fps'],3),
                    'source_timecode':timecode(n/manifest['video_meta']['fps']),
                    'source_selection':'shared_automatic_CC_candidate_keyframe',
                    'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
                scene['frames'].append(fr);changed=True
            scene['frames'].sort(key=lambda f:f['source_frame_index'])
        if changed:
            manifest['event_focus_frame_policy']='Automatic CC candidate keyframes added identically for both arms; human labels never read'
            dump(base/'manifest.json',manifest)
    finally:
        if cap is not None:cap.release()
    return manifest
