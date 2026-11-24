"""Time sync without OCR: Gemini reads the HUD clock from top-of-screen crops.

Used for the OCR-removed comparison and for the OCR vs VLM HUD reading check.
Produces the same {'segments': [...]} shape as vod.sync.build_sync, with a single segment.
"""
from __future__ import annotations

import re
import statistics
import time
from pathlib import Path

from pipeline.core import dump, load, video_meta
from .timesync import mmss

PROMPT = '''리그 오브 레전드 경기 화면의 위쪽 부분 이미지들입니다. 각 이미지에서 게임 시계(경기 경과 시간, mm:ss)와 위쪽 가운데 팀 킬 점수를 읽으세요.
보이지 않거나 확실하지 않으면 null로 두세요. 추측하지 마세요.
JSON만 반환: {"readings":[{"image_id":"c01","game_clock":"12:34","blue_kills":3,"red_kills":5,"confidence":"high|medium|low"}]}
'''


def parse_clock(text):
    m = re.fullmatch(r'\s*(\d{1,2}):([0-5]\d)\s*', str(text or ''))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _crops(video: Path, times: list, dest: Path, band: float = 0.14) -> list:
    import cv2
    dest.mkdir(parents=True, exist_ok=True)
    meta = video_meta(video)
    cap = cv2.VideoCapture(str(video))
    out = []
    try:
        for i, t in enumerate(times, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(meta['frame_count'] - 1, int(round(t * meta['fps']))))
            ok, frame = cap.read()
            if not ok:
                continue
            path = dest / f'c{i:02d}.jpg'
            cv2.imwrite(str(path), frame[:max(40, int(frame.shape[0] * band))], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            out.append({'image_id': f'c{i:02d}', 'video_sec': round(t, 2), 'path': path})
    finally:
        cap.release()
    return out


def read_hud(video: Path, times: list, dest: Path, model: str):
    """One Gemini request for all times. Returns (readings, reply)."""
    from RAG.coaching import _coaching_request
    crops = _crops(video, times, dest)
    if not crops:
        raise RuntimeError('could not read frames for clock crops')
    prompt = PROMPT + '\n'.join(f"IMAGE {c['image_id']}: 영상 {mmss(c['video_sec'])}" for c in crops)
    reply = _coaching_request(model, prompt, [('clock', {'frame_id': c['image_id'], 'source_timecode': mmss(c['video_sec'])},
                                               c['path']) for c in crops])
    by_id = {r.get('image_id'): r for r in reply['parsed'].get('readings', []) if isinstance(r, dict)}
    readings = []
    for c in crops:
        r = by_id.get(c['image_id'], {})
        readings.append({'video_sec': c['video_sec'], 'game_clock': r.get('game_clock'),
                         'game_sec': parse_clock(r.get('game_clock')), 'blue_kills': r.get('blue_kills'),
                         'red_kills': r.get('red_kills'), 'confidence': r.get('confidence')})
    return readings, reply


def vlm_sync(video: Path, work: Path, model: str, samples: int = 8) -> dict:
    """Median offset of the clock readings (outliers over 2 s removed). Cached in work/sync_vlm.json."""
    cache = work / 'sync_vlm.json'
    if cache.is_file():
        return load(cache)
    started = time.monotonic()
    duration = video_meta(video)['duration_sec']
    times = [duration * (0.06 + 0.88 * i / max(1, samples - 1)) for i in range(samples)]
    readings, reply = read_hud(video, times, work / 'clock_crops', model)
    offsets = [r['video_sec'] - r['game_sec'] for r in readings if r['game_sec'] is not None]
    if len(offsets) < 2:
        raise RuntimeError('VLM read fewer than 2 clocks')
    mid = statistics.median(offsets)
    kept = [o for o in offsets if abs(o - mid) <= 2.0]
    result = {'method': 'vlm',
              'segments': [{'video_start': 0.0, 'video_end': duration, 'game_start': -1e9, 'game_end': 1e9,
                            'offset': round(statistics.median(kept), 2), 'points': len(kept)}],
              'clock_read_rate': round(len(offsets) / len(readings), 3), 'readings': readings,
              'elapsed_sec': round(time.monotonic() - started, 2), 'api_calls': 1,
              'token_usage': reply['token_usage'],
              'note': 'single segment: pauses and speed changes are not detected'}
    dump(cache, result)
    return result
