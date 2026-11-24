from vod.sync import video_to_game

# 킬 스코어 변화로 영상 내 킬 이벤트를 검출한다 (Riot 타임라인이 없어도 동작).
# OCR 흔들림을 막기 위해 같은 값이 STABLE_READS번 연속 읽혀야 확정한다.

STABLE_READS = 2
MAX_JUMP = 5  # 한 번에 이만큼 넘게 오르면 오인식으로 본다


def detect_kills(rows, sync=None):
    events = []
    for team in ('blue', 'red'):
        confirmed = None
        candidate, count, first_sec = None, 0, None
        for row in rows:
            value = row.get(team)
            if value is None:
                continue
            if value == candidate:
                count += 1
            else:
                candidate, count, first_sec = value, 1, row['video_sec']
            if count < STABLE_READS or candidate == confirmed:
                continue
            if confirmed is None:
                confirmed = candidate
                continue
            diff = candidate - confirmed
            if 0 < diff <= MAX_JUMP:
                events.append({
                    'source': 'video_score',
                    'type': 'CHAMPION_KILL',
                    'team': team,
                    'kills': diff,
                    'video_sec': first_sec,
                    'game_sec': video_to_game(sync, first_sec) if sync else None,
                    'score_after': candidate,
                })
                confirmed = candidate
            elif diff < 0 and count >= STABLE_READS * 3:
                # 리플레이를 되감은 경우: 낮아진 값을 새 기준으로 받아들임
                confirmed = candidate
    events.sort(key=lambda e: e['video_sec'])
    return events
