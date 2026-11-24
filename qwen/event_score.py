import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TIMELINE_DIR = ROOT / 'data' / 'riot' / 'timelines'
OUTPUT_DIR = ROOT / 'results' / 'events'

# These are experimental ranking weights, not an official Riot scoring formula.
EVENT_WEIGHTS = {
    'CHAMPION_KILL': 5.0,
    'ELITE_MONSTER_KILL': 7.0,
    'BUILDING_KILL': 6.0,
    'TURRET_PLATE_DESTROYED': 1.0,
}
SPECIAL_BONUSES = {
    'KILL_FIRST_BLOOD': 3.0,
    'KILL_MULTI': 2.0,
    'KILL_ACE': 3.0,
}

# Group nearby events only if they are close in BOTH time and position.
GROUP_GAP_MS = 20_000
GROUP_DISTANCE = 2_500
CLIP_PADDING_BEFORE_MS = 12_000
CLIP_PADDING_AFTER_MS = 8_000
TOP_K = 10


def time_label(ms):
    seconds = max(0, int(ms) // 1000)
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def event_score(event):
    event_type = event.get('type')
    score = EVENT_WEIGHTS.get(event_type, 0.0)
    if event_type == 'CHAMPION_KILL':
        if event.get('shutdownBounty', 0) > 0:
            score += 2.0
        if len(event.get('assistingParticipantIds') or []) >= 2:
            score += 1.0
    elif event_type == 'ELITE_MONSTER_KILL':
        monster = event.get('monsterType')
        if monster in ('BARON_NASHOR', 'BARON'):
            score += 6.0
        elif monster == 'DRAGON':
            score += 3.0
        elif monster == 'RIFTHERALD':
            score += 2.0
        elif monster == 'HORDE':
            score = 2.0
    elif event_type == 'BUILDING_KILL':
        if event.get('buildingType') == 'INHIBITOR_BUILDING':
            score += 2.0
    return score


def extract_events(timeline):
    events = []
    for frame in timeline.get('info', {}).get('frames', []):
        for event in frame.get('events', []):
            typ = event.get('type')
            if typ not in EVENT_WEIGHTS:
                continue
            position = event.get('position')
            if not isinstance(position, dict) or not all(k in position for k in ('x', 'y')):
                position = None
            events.append({
                'timestamp_ms': int(event.get('timestamp', 0)),
                'type': typ,
                'score': event_score(event),
                'position': position,
                'killer_id': event.get('killerId'),
                'victim_id': event.get('victimId'),
                'assisting_ids': event.get('assistingParticipantIds', []),
                'monster_type': event.get('monsterType'),
                'monster_subtype': event.get('monsterSubType'),
                'building_type': event.get('buildingType'),
            })
    events.sort(key=lambda e: e['timestamp_ms'])
    return events


def nearby_position(a, b):
    if a is None or b is None:
        return False
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    return dx * dx + dy * dy <= GROUP_DISTANCE * GROUP_DISTANCE


def make_groups(events):
    # Each event joins the most recent spatially close group within the time limit.
    groups = []
    for event in events:
        candidates = []
        for group in groups:
            if event['timestamp_ms'] - group['end_ms'] > GROUP_GAP_MS:
                continue
            if nearby_position(event['position'], group['last_position']):
                candidates.append(group)
        if candidates:
            group = max(candidates, key=lambda g: g['end_ms'])
        else:
            group = {'start_ms': event['timestamp_ms'], 'end_ms': event['timestamp_ms'],
                     'last_position': event['position'], 'events': []}
            groups.append(group)
        group['events'].append(event)
        group['end_ms'] = event['timestamp_ms']
        if event['position'] is not None:
            group['last_position'] = event['position']
    return groups


def serialize_groups(groups, timeline):
    end_ms = max((f.get('timestamp', 0) for f in timeline.get('info', {}).get('frames', [])), default=0)
    output = []
    for group in groups:
        event_counts = Counter(event['type'] for event in group['events'])
        score = sum(event['score'] for event in group['events'])
        # Avoid double counting: SPECIAL_KILL is a bonus annotation on a kill, not an extra kill.
        # Group coordinates come from event data; video clip times are only provisional.
        output.append({
            'start_ms': group['start_ms'],
            'end_ms': group['end_ms'],
            'time': f"{time_label(group['start_ms'])}~{time_label(group['end_ms'])}",
            'clip_start_ms': max(0, group['start_ms'] - CLIP_PADDING_BEFORE_MS),
            'clip_end_ms': min(end_ms, group['end_ms'] + CLIP_PADDING_AFTER_MS),
            'score': round(score, 2),
            'event_counts': dict(event_counts),
            'events': group['events'],
        })
    output.sort(key=lambda g: (-g['score'], g['start_ms']))
    for i, group in enumerate(output, 1):
        group['rank'] = i
    return output


def analyze_file(path):
    with path.open(encoding='utf-8') as f:
        timeline = json.load(f)
    match_id = timeline.get('metadata', {}).get('matchId', path.stem)
    events = extract_events(timeline)
    groups = serialize_groups(make_groups(events), timeline)
    result = {
        'match_id': match_id,
        'note': 'Heuristic scores; not an official Riot metric. Clip times must be aligned with the VOD.',
        'event_count': len(events),
        'group_count': len(groups),
        'top_groups': groups[:TOP_K],
        'all_groups': groups,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'{match_id}_scored.json'
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n{match_id}: {len(events)} events, {len(groups)} groups -> {output_path.name}')
    for g in groups[:5]:
        print(f"  #{g['rank']} {g['time']} score={g['score']:.1f} {g['event_counts']}")


def main():
    if not TIMELINE_DIR.is_dir():
        print(f'Timeline directory not found: {TIMELINE_DIR}')
        return
    paths = sorted(TIMELINE_DIR.glob('*.json'))
    if not paths:
        print('No timeline JSON files found.')
        return
    print(f'Analyzing {len(paths)} timeline file(s).')
    for path in paths:
        try:
            analyze_file(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f'Failed to analyze {path.name}: {exc}')


if __name__ == '__main__':
    main()
