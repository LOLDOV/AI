import statistics

# 시간 동기화 평가: Riot 타임라인의 킬 시각과 영상 HUD에서 검출한 킬 시각(게임 시간 변환)을 대조.
# participantId 1~5는 블루(100), 6~10은 레드(200) 팀.

MATCH_WINDOW_SEC = 10


def riot_kills(timeline):
    kills = []
    for frame in timeline.get('info', {}).get('frames', []):
        for event in frame.get('events', []):
            if event.get('type') != 'CHAMPION_KILL':
                continue
            killer = event.get('killerId') or 0
            victim = event.get('victimId') or 0
            # 처형(killerId 0)은 피해자 반대 팀 점수로 집계된다
            team = 'blue' if 1 <= killer <= 5 else 'red' if killer >= 6 else ('red' if victim <= 5 else 'blue')
            kills.append({'game_sec': event.get('timestamp', 0) / 1000, 'team': team})
    return kills


def compare_kills(video_kills, timeline):
    reference = riot_kills(timeline)
    detected = []
    for k in video_kills:
        if k.get('game_sec') is None:
            continue
        detected.extend([{'game_sec': k['game_sec'], 'team': k['team']}] * k['kills'])

    used = set()
    errors = []
    for ref in reference:
        best = None
        for i, det in enumerate(detected):
            if i in used or det['team'] != ref['team']:
                continue
            diff = det['game_sec'] - ref['game_sec']
            if abs(diff) <= MATCH_WINDOW_SEC and (best is None or abs(diff) < abs(best[1])):
                best = (i, diff)
        if best:
            used.add(best[0])
            errors.append(best[1])

    abs_errors = [abs(e) for e in errors]
    return {
        'riot_kills': len(reference),
        'video_kills': len(detected),
        'matched': len(errors),
        'recall': round(len(errors) / len(reference), 3) if reference else None,
        'precision': round(len(errors) / len(detected), 3) if detected else None,
        'signed_error_median': round(statistics.median(errors), 2) if errors else None,
        'abs_error_median': round(statistics.median(abs_errors), 2) if errors else None,
        'abs_error_max': round(max(abs_errors), 2) if errors else None,
    }
