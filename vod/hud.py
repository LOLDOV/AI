import re

import cv2
from rapidocr_onnxruntime import RapidOCR

# 상단 HUD(게임 시계·킬 스코어) 인식.
# 시계 위치를 한 번 찾아두고(calibrate), 스코어 영역은 시계 글자 높이(h) 기준 상대 좌표로 자른다.
# 상대 좌표는 1282x832 리플레이 녹화본에서 측정한 값.

CLOCK_RE = re.compile(r'(\d{1,2})\s*[:.]\s*(\d{2})')
DIGITS_RE = re.compile(r'\d+')

# (x0, y0, x1, y1), 시계 중심 기준, 시계 글자 높이 단위
BLUE_SCORE_BOX = (-4.2, -4.1, -0.5, -1.65)
RED_SCORE_BOX = (0.5, -4.1, 4.7, -1.65)
CLOCK_BOX = (-1.9, -0.9, 1.9, 0.9)

UPSCALE = 2


class HudReader:
    def __init__(self):
        self.ocr = RapidOCR()
        self.clock = None  # (cx, cy, h)

    def calibrate(self, frame):
        """Find the game clock in the top-center strip. Returns True on success."""
        h, w = frame.shape[:2]
        x0 = int(w * 0.3)
        strip = frame[0:int(h * 0.2), x0:int(w * 0.7)]
        result, _ = self.ocr(strip)
        for box, text, conf in result or []:
            if not CLOCK_RE.fullmatch(text.strip()) or conf < 0.6:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            self.clock = (
                x0 + (min(xs) + max(xs)) / 2,
                (min(ys) + max(ys)) / 2,
                max(ys) - min(ys),
            )
            return True
        return False

    def _crop(self, frame, rel_box):
        cx, cy, ch = self.clock
        x0, y0, x1, y1 = rel_box
        h, w = frame.shape[:2]
        left = max(0, int(cx + x0 * ch))
        right = min(w, int(cx + x1 * ch))
        top = max(0, int(cy + y0 * ch))
        bottom = min(h, int(cy + y1 * ch))
        crop = frame[top:bottom, left:right]
        return cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    def _read(self, crop):
        result, _ = self.ocr(crop, use_det=False, use_cls=False)
        if not result:
            return '', 0.0
        text, conf = result[0]
        return text.strip(), float(conf)

    def read_clock(self, frame):
        """Game time in seconds, or None."""
        text, conf = self._read(self._crop(frame, CLOCK_BOX))
        match = CLOCK_RE.search(text)
        if not match or conf < 0.6:
            return None
        minutes, seconds = int(match.group(1)), int(match.group(2))
        if seconds >= 60:
            return None
        return minutes * 60 + seconds

    def read_score(self, frame):
        """(blue_kills, red_kills); each may be None if unreadable."""
        scores = []
        for rel_box in (BLUE_SCORE_BOX, RED_SCORE_BOX):
            text, conf = self._read(self._crop(frame, rel_box))
            digits = DIGITS_RE.findall(text)
            scores.append(int(digits[0]) if digits and conf >= 0.5 and len(digits[0]) <= 3 else None)
        return tuple(scores)

    def score_crops(self, frame):
        return [self._crop(frame, box) for box in (BLUE_SCORE_BOX, RED_SCORE_BOX)]
