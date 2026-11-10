import statistics

# 게임 시간 ↔ 영상 시간 동기화.
# HUD 시계 OCR 값으로 offset(= video_sec - game_sec)을 구하고,
# 리플레이 일시정지·배속·건너뛰기로 offset이 바뀌는 지점마다 구간을 나눈다.

SEGMENT_TOLERANCE = 2.0   # 같은 구간으로 볼 offset 차이(초)
MIN_SEGMENT_POINTS = 5    # 이보다 짧은 구간은 OCR 오인식으로 보고 버림


def clock_points(rows):
    return [(r['video_sec'], r['game_sec']) for r in rows if r.get('game_sec') is not None]


def build_segments(points):
    segments = []
    current = []
    for video_sec, game_sec in points:
        offset = video_sec - game_sec
        if current:
            ref = statistics.median(v - g for v, g in current[-MIN_SEGMENT_POINTS:])
            if abs(offset - ref) > SEGMENT_TOLERANCE:
                segments.append(current)
                current = []
        current.append((video_sec, game_sec))
    if current:
        segments.append(current)

    # 짧은 구간(OCR 오인식)을 버린 뒤, offset이 같은 인접 구간은 다시 합친다
    kept = []
    for seg in segments:
        if len(seg) < MIN_SEGMENT_POINTS:
            continue
        offset = statistics.median(v - g for v, g in seg)
        if kept and abs(offset - statistics.median(v - g for v, g in kept[-1])) <= SEGMENT_TOLERANCE:
            kept[-1].extend(seg)
        else:
            kept.append(seg)
    segments = kept

    result = []
    for seg in segments:
        if len(seg) < MIN_SEGMENT_POINTS:
            continue
        offsets = [v - g for v, g in seg]
        offset = statistics.median(offsets)
        residuals = [abs(o - offset) for o in offsets]
        result.append({
            'video_start': seg[0][0],
            'video_end': seg[-1][0],
            'game_start': seg[0][1],
            'game_end': seg[-1][1],
            'offset': round(offset, 2),
            'points': len(seg),
            'residual_median': round(statistics.median(residuals), 3),
            'residual_max': round(max(residuals), 3),
        })
    return result


def build_sync(rows):
    points = clock_points(rows)
    segments = build_segments(points)
    total = len(rows)
    return {
        'clock_read_rate': round(len(points) / total, 3) if total else 0.0,
        'segments': segments,
    }


def game_to_video(sync, game_sec):
    """Map game time to video time; None if the game time is not covered by the VOD."""
    for seg in sync['segments']:
        if seg['game_start'] - 1 <= game_sec <= seg['game_end'] + 1:
            return game_sec + seg['offset']
    return None


def video_to_game(sync, video_sec):
    for seg in sync['segments']:
        if seg['video_start'] - 1 <= video_sec <= seg['video_end'] + 1:
            return video_sec - seg['offset']
    return None
