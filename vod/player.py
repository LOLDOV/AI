from collections import Counter

import cv2

from vod.video import frame_at

# 분석 대상 플레이어의 팀·슬롯 판별.
# 리플레이 좌하단의 큰 초상화(선택된 챔피언)를 좌측(블루)·우측(레드) 팀 초상화 열과 템플릿 매칭한다.
# 좌표는 1282x832 녹화본 기준 비율이며, 시청자가 다른 플레이어로 바꿀 수 있어 여러 시점에서 투표한다.

REF_W, REF_H = 1282, 832
BIG_PORTRAIT = (14, 663, 46, 693)     # 레벨 숫자를 피한 안쪽 영역
LEFT_COLUMN = (12, 125, 62, 505)
RIGHT_COLUMN = (1220, 125, 1270, 505)
SIDE_SCALES = (0.5, 0.55, 0.6, 0.65)  # 큰 초상화 대비 옆 초상화 크기
MIN_SCORE = 0.55
SAMPLE_SECONDS = (90, 240, 420, 600, 780, 960)


def scaled_box(frame, box):
    h, w = frame.shape[:2]
    sx, sy = w / REF_W, h / REF_H
    x0, y0, x1, y1 = box
    return frame[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]


def match_side(frame):
    """Returns (team, slot, score) for one frame or None."""
    big = cv2.cvtColor(scaled_box(frame, BIG_PORTRAIT), cv2.COLOR_BGR2GRAY)
    best = None
    for team, column_box in (('blue', LEFT_COLUMN), ('red', RIGHT_COLUMN)):
        column = cv2.cvtColor(scaled_box(frame, column_box), cv2.COLOR_BGR2GRAY)
        for scale in SIDE_SCALES:
            tmpl = cv2.resize(big, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if tmpl.shape[0] >= column.shape[0] or tmpl.shape[1] >= column.shape[1]:
                continue
            result = cv2.matchTemplate(column, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if best is None or score > best[2]:
                slot = min(4, int(loc[1] / column.shape[0] * 5))
                best = (team, slot, float(score))
    return best if best and best[2] >= MIN_SCORE else None


def detect_player(video_path, duration_sec):
    votes = []
    for sec in SAMPLE_SECONDS:
        if sec >= duration_sec - 5:
            break
        frame = frame_at(video_path, sec)
        if frame is None:
            continue
        found = match_side(frame)
        if found:
            votes.append(found)
    if not votes:
        return {'team': None, 'slot': None, 'confidence': 0.0, 'samples': 0}
    (team, slot), count = Counter((t, s) for t, s, _ in votes).most_common(1)[0]
    return {
        'team': team,
        'slot': slot,
        'confidence': round(count / len(votes), 2),
        'samples': len(votes),
        'mean_score': round(sum(v[2] for v in votes if (v[0], v[1]) == (team, slot)) / count, 3),
    }
