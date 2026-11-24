"""Conservative, optional HUD clock OCR. No fabricated zero-offset fallback.

A clock anchor proves a consistent overlay timer, NOT that the selected Match ID
belongs to the MP4. Match identity remains user-selected and independently guarded.
"""
from __future__ import annotations
import os
import re
from statistics import median
from pathlib import Path

CLOCK = re.compile(r'(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):([0-5]\d)(?!\d)')


def parse_clock_strings(text: str) -> list[int]:
    out = []
    for h, m, s in CLOCK.findall(text):
        if int(m) >= 60 and h:
            continue
        total = int(h or 0) * 3600 + int(m) * 60 + int(s)
        if 10 <= total <= 4 * 3600:
            out.append(total)
    return out


def consensus_offset(samples: list[dict], *, minimum: int = 3, tolerance: float = 1.8) -> dict:
    """Require 3+ stable same-ROI readings spread across the VOD."""
    groups = {}
    for sample in samples:
        try:
            video = float(sample['video_sec']); game = float(sample['game_sec'])
            group = str(sample['roi'])
            if 0 <= video <= 5 * 3600 and 0 <= game <= 5 * 3600:
                groups.setdefault(group, []).append((video, game, video-game))
        except (KeyError, TypeError, ValueError):
            continue
    best = None
    for roi, items in groups.items():
        if len(items) < minimum:
            continue
        offsets = [item[2] for item in items]
        offset = median(offsets)
        good = [item for item in items if abs(item[2]-offset) <= tolerance]
        if len(good) < minimum:
            continue
        if max(item[0] for item in good) - min(item[0] for item in good) < 60:
            continue
        # An extreme offset is more likely an unrelated timer; edited videos need review.
        offset = median(item[2] for item in good)
        if abs(offset) > 240:
            continue
        spread = max(abs(item[2]-offset) for item in good)
        proposed = {'status': 'ocr_candidate_not_riot_event_verified',
                    'offset_sec': round(offset, 2), 'roi': roi,
                    'readings': len(good), 'max_deviation_sec': round(spread, 2),
                    'samples': [{'video_sec': round(x, 2), 'game_sec': y} for x,y,_ in good]}
        if best is None or len(good) > best['readings'] or (
            len(good) == best['readings'] and spread < best['max_deviation_sec']
        ):
            best = proposed
    return best or {'status': 'unavailable', 'reason': '신뢰할 수 있는 공통 게임 시각 3개 이상을 찾지 못했습니다'}


def _default_rois() -> list[tuple[str, tuple[float,float,float,float]]]:
    raw = os.getenv('LOLDOV_HUD_TIMER_ROI', '').strip()
    if raw:
        try:
            coords = tuple(float(part) for part in raw.split(','))
            if len(coords) != 4 or any(x < 0 or x > 1 for x in coords) or coords[2] <= 0 or coords[3] <= 0 or coords[0]+coords[2]>1 or coords[1]+coords[3]>1:
                raise ValueError
            return [('configured', coords)]
        except ValueError as exc:
            raise ValueError('LOLDOV_HUD_TIMER_ROI는 x,y,w,h 형식의 0~1 비율값') from exc
    # Generic LoL HUD places can differ wildly: these are candidate regions, NOT guarantees.
    return [('topright', (.71, .00, .28, .15)), ('topmiddle', (.38,.00,.25,.14)),
            ('bottomright', (.70,.86,.29,.14))]


def estimate_from_video(video_path: str | Path) -> dict:
    try:
        import cv2, pytesseract
        from pipeline.core import video_meta
    except ImportError:
        return {'status':'unavailable','reason':'OCR 선택 의존성(cv2/pytesseract)을 찾지 못했습니다'}
    tesseract_cmd = os.getenv('TESSERACT_CMD', '').strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        meta = video_meta(Path(video_path))
    except Exception as exc:
        return {'status':'unavailable','reason':f'영상 열기 실패: {exc}'}
    cap = cv2.VideoCapture(str(video_path))
    samples = []
    duration = meta['duration_sec']
    times = [duration*x for x in (.16,.30,.45,.59,.74,.87)]
    rois = _default_rois()
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h,w = frame.shape[:2]
            for roi_id,(x,y,rw,rh) in rois:
                crop = frame[int(y*h):int((y+rh)*h), int(x*w):int((x+rw)*w)]
                if crop.size == 0:
                    continue
                gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray,None,fx=2.5,fy=2.5)
                gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                try:
                    text = pytesseract.image_to_string(gray, config='--psm 6')
                except (OSError, RuntimeError, pytesseract.TesseractNotFoundError):
                    return {'status':'unavailable','reason':'Tesseract 실행파일 미설치 또는 경로 설정 필요'}
                possibilities = parse_clock_strings(text)
                # Ambiguous OCR tokens are not trusted as anchors.
                if len(possibilities) == 1:
                    samples.append({'roi':roi_id,'video_sec':t,'game_sec':possibilities[0]})
    finally:
        cap.release()
    output = consensus_offset(samples)
    output['ocr_samples_examined'] = len(times)
    output['ocr_candidate_readings'] = len(samples)
    return output
