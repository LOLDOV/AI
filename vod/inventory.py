import csv
import sys
from pathlib import Path

from vod.hud import HudReader
from vod.video import frame_at, probe

# 수집한 VOD의 수량·분포 집계 (해상도, 길이, 용량, HUD 시계 인식 여부)
# 예) python -m vod.inventory data/vod

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'results' / 'vod_inventory.csv'
PROBE_SECONDS = (120, 300, 600)


def inspect(path, hud):
    info = probe(path)
    info['clock_found'] = False
    info['offset_sec'] = None
    for sec in PROBE_SECONDS:
        if sec >= info['duration_sec']:
            break
        frame = frame_at(path, sec)
        if frame is None:
            continue
        hud.clock = None
        if hud.calibrate(frame):
            game_sec = hud.read_clock(frame)
            info['clock_found'] = True
            info['offset_sec'] = None if game_sec is None else sec - game_sec
            break
    return info


def main():
    vod_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'data' / 'vod'
    paths = sorted(p for p in vod_dir.iterdir() if p.suffix.lower() in ('.mp4', '.webm', '.mkv'))
    hud = HudReader()
    rows = []
    for path in paths:
        try:
            row = inspect(path, hud)
        except Exception as exc:
            row = {'file': path.name, 'error': str(exc)}
        rows.append(row)
        print(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ['file', 'width', 'height', 'fps', 'codec', 'duration_sec', 'size_mb', 'clock_found', 'offset_sec', 'error']
    with OUTPUT.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if 'error' not in r]
    if ok:
        total_min = sum(r['duration_sec'] for r in ok) / 60
        resolutions = sorted({f"{r['width']}x{r['height']}" for r in ok})
        print(f'\n{len(ok)} videos, {total_min:.1f} min total, resolutions: {", ".join(resolutions)}')
    print(f'Saved: {OUTPUT}')


if __name__ == '__main__':
    main()
