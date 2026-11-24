"""Video time <-> game time using the vod/ HUD OCR scan."""
from __future__ import annotations

from pathlib import Path

from pipeline.core import dump, load
from vod.sync import build_sync, game_to_video


def mmss(sec):
    if sec is None:
        return None
    sec = max(0, int(round(sec)))
    return f'{sec // 60:02d}:{sec % 60:02d}'


def scan(video: Path, cache: Path, rescan: bool = False) -> dict:
    """Run vod.scan once per video and reuse the saved result."""
    if cache.is_file() and not rescan:
        return load(cache)
    from vod.scan import scan_video
    data = scan_video(video, log_every=0)
    dump(cache, data)
    return data


def ocr_sync(video: Path, cache: Path, rescan: bool = False):
    """Return (sync, rows). sync['segments'] splits pauses and speed changes."""
    data = scan(video, cache, rescan)
    if data.get('clock_box') is None:
        raise RuntimeError('game clock not found on screen')
    sync = build_sync(data['rows'])
    if not sync['segments']:
        raise RuntimeError('no consistent clock segment')
    return sync, data['rows']


def to_video(sync: dict, game_sec: float, duration: float):
    """Game time -> video time, or None if that moment is not in the video."""
    t = game_to_video(sync, game_sec)
    if t is None or t < 0 or t > duration:
        return None
    return round(t, 2)


def hud_at(rows: list, video_sec: float, tol: float = 1.5):
    """OCR clock and team kills at the nearest 1-second sample."""
    best = min(rows, key=lambda r: abs(r['video_sec'] - video_sec), default=None)
    if not best or abs(best['video_sec'] - video_sec) > tol:
        return None
    return {'video_sec': best['video_sec'], 'game_clock': mmss(best.get('game_sec')),
            'blue_kills': best.get('blue'), 'red_kills': best.get('red')}
