import cv2
import numpy as np

from vod.hud import HudReader
from vod.video import iter_frames, probe

# 영상을 한 번 훑으며 초 단위 신호를 기록한다.
#  - game_sec: HUD 시계 OCR (시간 동기화용)
#  - blue/red: 킬 스코어 OCR (이벤트 데이터 없이도 킬 시점 검출)
#  - motion: 화면 변화량, red_bars/blue_bars: 체력바 색 픽셀 비율 (교전·챔피언 밀집 근사)
# 체력바 기반 신호는 휴리스틱이며 개발 데이터에서 임계값을 조정해야 한다.

SAMPLE_FPS = 1.0
CALIBRATION_TRIES = 30

# 화면 가장자리 HUD를 제외한 플레이 영역 (비율)
PLAY_AREA = (0.06, 0.13, 0.84, 0.75)  # x0, y0, x1, y1

RED_RANGES = [((0, 120, 120), (8, 255, 255)), ((170, 120, 120), (180, 255, 255))]
BLUE_RANGES = [((95, 120, 140), (108, 255, 255))]


def play_area(frame):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = PLAY_AREA
    return frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def bar_ratio(hsv, ranges, kernel):
    mask = None
    for low, high in ranges:
        part = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = part if mask is None else mask | part
    # 가로로 긴 막대 모양만 남긴다
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return float(np.count_nonzero(mask)) / mask.size


def scan_video(path, sample_fps=SAMPLE_FPS, log_every=120, progress=None):
    """progress: optional callable(fraction 0~1)."""
    duration = probe(path)['duration_sec']
    hud = HudReader()
    rows = []
    prev_small = None
    kernel = None

    for video_sec, frame in iter_frames(path, sample_fps):
        if hud.clock is None and len(rows) < CALIBRATION_TRIES * sample_fps:
            hud.calibrate(frame)

        area = play_area(frame)
        if kernel is None:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(5, area.shape[1] // 80), 1)
            )
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        small = cv2.cvtColor(cv2.resize(area, (160, 90)), cv2.COLOR_BGR2GRAY)
        motion = 0.0 if prev_small is None else float(
            np.mean(cv2.absdiff(small, prev_small))
        )
        prev_small = small

        row = {
            'video_sec': round(video_sec, 2),
            'game_sec': None,
            'blue': None,
            'red': None,
            'motion': round(motion, 3),
            'red_bars': round(bar_ratio(hsv, RED_RANGES, kernel), 5),
            'blue_bars': round(bar_ratio(hsv, BLUE_RANGES, kernel), 5),
        }
        if hud.clock is not None:
            row['game_sec'] = hud.read_clock(frame)
            row['blue'], row['red'] = hud.read_score(frame)
        rows.append(row)

        if progress and len(rows) % 10 == 0:
            progress(min(1.0, video_sec / duration))
        if log_every and len(rows) % log_every == 0:
            print(f'  scanned {video_sec:.0f}s (clock={row["game_sec"]}, score={row["blue"]}:{row["red"]})')

    return {
        'clock_box': hud.clock,
        'sample_fps': sample_fps,
        'rows': rows,
    }
