import numpy as np

from vod.sync import game_to_video, video_to_game

# 주요 장면 후보 선정.
#  1) 이벤트 기반: 영상 킬 스코어 변화 + (있으면) Riot 타임라인 이벤트 그룹
#  2) 이벤트 없는 장면 보완: 교전 신호(체력바 밀집·화면 변화)가 높은 구간
# 겹치는 구간은 병합하고, 구간마다 대표 프레임 시각을 고른다.
# 가중치·구간 길이는 실험값이며 개발 데이터로 조정한 뒤 테스트 전에 고정한다.

PADDING_BEFORE = 12.0
PADDING_AFTER = 8.0
KILL_SCORE = 5.0
MULTI_KILL_BONUS = 2.0
VISUAL_SCORE = 3.0

SMOOTH_SEC = 5
VISUAL_PERCENTILE = 92
VISUAL_MIN_GAP = 20.0
VISUAL_MIN_GAME_SEC = 90   # 초반 우물·라인 이동 구간 제외
FRAMES_PER_SCENE = 6
TOP_K = 10


def intensity_series(rows):
    combat = np.array([np.sqrt(r['red_bars'] * r['blue_bars']) for r in rows])
    motion = np.array([r['motion'] for r in rows])

    def norm(a):
        ref = np.percentile(a, 95) or 1.0
        return np.clip(a / ref, 0, 2)

    raw = 0.7 * norm(combat) + 0.3 * norm(motion)
    kernel = np.ones(SMOOTH_SEC) / SMOOTH_SEC
    return np.convolve(raw, kernel, mode='same')


def kill_candidates(kills):
    return [{
        'start': k['video_sec'] - PADDING_BEFORE,
        'end': k['video_sec'] + PADDING_AFTER,
        'score': KILL_SCORE * k['kills'] + (MULTI_KILL_BONUS if k['kills'] > 1 else 0),
        'sources': ['video_score'],
        'events': [k],
    } for k in kills]


def timeline_candidates(groups, sync):
    """Riot timeline groups (qwen/event_score.py output) mapped onto video time."""
    result = []
    for group in groups:
        start = game_to_video(sync, group['clip_start_ms'] / 1000)
        end = game_to_video(sync, group['clip_end_ms'] / 1000)
        if start is None or end is None:
            continue
        result.append({
            'start': start,
            'end': end,
            'score': group['score'],
            'sources': ['riot_timeline'],
            'events': [{'source': 'riot_timeline', **e} for e in group['events']],
        })
    return result


def overlaps(a, b):
    return a['start'] < b['end'] and b['start'] < a['end']


def visual_candidates(rows, intensity, existing):
    threshold = np.percentile(intensity, VISUAL_PERCENTILE)
    peaks = []
    for i in range(1, len(intensity) - 1):
        if intensity[i] < threshold:
            continue
        if intensity[i] < intensity[i - 1] or intensity[i] < intensity[i + 1]:
            continue
        game_sec = rows[i].get('game_sec')
        if game_sec is not None and game_sec < VISUAL_MIN_GAME_SEC:
            continue
        peaks.append(i)

    result = []
    for i in sorted(peaks, key=lambda i: -intensity[i]):
        t = rows[i]['video_sec']
        cand = {
            'start': t - PADDING_BEFORE,
            'end': t + PADDING_AFTER,
            'score': round(VISUAL_SCORE * float(intensity[i]), 2),
            'sources': ['visual'],
            'events': [{'source': 'visual', 'type': 'COMBAT_SIGNAL',
                        'video_sec': t, 'intensity': round(float(intensity[i]), 3)}],
        }
        if any(abs(t - c['events'][0]['video_sec']) < VISUAL_MIN_GAP for c in result):
            continue
        if any(overlaps(cand, c) for c in existing):
            continue
        result.append(cand)
    return result


def merge(candidates):
    merged = []
    for cand in sorted(candidates, key=lambda c: c['start']):
        if merged and cand['start'] <= merged[-1]['end']:
            last = merged[-1]
            last['end'] = max(last['end'], cand['end'])
            last['score'] += cand['score']
            last['sources'] = sorted(set(last['sources']) | set(cand['sources']))
            last['events'].extend(cand['events'])
        else:
            merged.append({**cand, 'events': list(cand['events'])})
    return merged


def pick_frames(scene, rows, intensity, n=FRAMES_PER_SCENE):
    """Evenly spaced timestamps plus the peak-intensity moment, in time order."""
    times = list(np.linspace(scene['start'], scene['end'], n - 1))
    inside = [i for i, r in enumerate(rows) if scene['start'] <= r['video_sec'] <= scene['end']]
    if inside:
        peak = max(inside, key=lambda i: intensity[i])
        times.append(rows[peak]['video_sec'])
    return sorted(round(float(t), 1) for t in set(times))


def time_label(sec):
    if sec is None:
        return None
    sec = max(0, int(round(sec)))
    return f'{sec // 60:02d}:{sec % 60:02d}'


def build_scenes(scan, sync, kills, timeline_groups=None, duration=None, top_k=TOP_K):
    rows = scan['rows']
    intensity = intensity_series(rows)

    event_cands = kill_candidates(kills)
    if timeline_groups:
        event_cands += timeline_candidates(timeline_groups, sync)
    event_cands = merge(event_cands)
    visual = visual_candidates(rows, intensity, event_cands)
    scenes = merge(event_cands + visual)

    end_limit = duration if duration is not None else rows[-1]['video_sec']
    for scene in scenes:
        scene['start'] = round(max(0.0, scene['start']), 1)
        scene['end'] = round(min(end_limit, scene['end']), 1)
        scene['score'] = round(scene['score'], 2)
        scene['event_free'] = scene['sources'] == ['visual']
        scene['game_time'] = (
            f"{time_label(video_to_game(sync, scene['start']))}~"
            f"{time_label(video_to_game(sync, scene['end']))}"
        )
        scene['frame_times'] = pick_frames(scene, rows, intensity)

    scenes.sort(key=lambda s: (-s['score'], s['start']))
    for i, scene in enumerate(scenes, 1):
        scene['rank'] = i
    return {
        'scene_count': len(scenes),
        'event_free_count': sum(s['event_free'] for s in scenes),
        'top_scenes': scenes[:top_k],
        'all_scenes': scenes,
    }
