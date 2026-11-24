import json
import os
from pathlib import Path

from vod.scenes import intensity_series

# results/ 아래 분석 결과를 모아 정적 웹 리포트(results/site/index.html)를 만든다.
# 데이터는 HTML에 내장하고, 프레임 이미지는 상대 경로(../<영상>/frames/...)로 참조한다.
# 예) python -m report.build_site  ->  cd results && python -m http.server 8000  ->  /site/
# HTML 틀은 Frontend 저장소의 report/template.html (기본: 같은 상위 폴더의 Frontend 클론, LOLDOV_FRONTEND_DIR 로 변경)

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / 'results'
SITE_DIR = RESULT_DIR / 'site'
FRONTEND_DIR = Path(os.getenv('LOLDOV_FRONTEND_DIR', ROOT.parent / 'Frontend'))
TEMPLATE = FRONTEND_DIR / 'report' / 'template.html'
SERIES_STEP = 2  # 차트용 신호는 2초 간격으로 줄인다


def read_json(path, default=None):
    if not path.is_file():
        return default
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def frame_url(path):
    return os.path.relpath(path, SITE_DIR).replace(os.sep, '/')


def video_payload(video_dir, url_for=frame_url):
    meta = read_json(video_dir / 'run_meta.json')
    scan = read_json(video_dir / 'scan.json')
    scenes = read_json(video_dir / 'scenes.json')
    if not (meta and scan and scenes):
        return None
    sync = read_json(video_dir / 'sync.json', {'segments': [], 'clock_read_rate': 0})
    kills = read_json(video_dir / 'kills.json', [])
    vlm = {v['rank']: v for v in read_json(video_dir / 'vlm.json', []) or []}

    rows = scan['rows']
    intensity = intensity_series(rows)
    offset = sync['segments'][0]['offset'] if sync['segments'] else 0
    series = []
    for i in range(0, len(rows), SERIES_STEP):
        r = rows[i]
        game_sec = r['game_sec'] if r['game_sec'] is not None else r['video_sec'] - offset
        series.append([round(game_sec, 1), round(float(intensity[i]), 3)])

    scene_list = []
    for s in sorted(scenes['all_scenes'], key=lambda s: s['rank']):
        item = {
            'rank': s['rank'],
            'start': s['start'] - offset,
            'end': s['end'] - offset,
            'video_start': s['start'],
            'game_time': s['game_time'],
            'score': s['score'],
            'sources': s['sources'],
            'event_free': s['event_free'],
            'events': [{k: e.get(k) for k in ('source', 'type', 'team', 'kills', 'game_sec', 'video_sec',
                                               'score_after', 'intensity', 'monster_type', 'building_type')}
                       for e in s['events']],
            'frames': [{'url': url_for(f['path']), 'game_time': f['game_time'], 'id': f['frame_id']}
                       for f in s.get('frames', [])],
        }
        v = vlm.get(s['rank'])
        if v:
            item['vlm'] = {
                'analysis': v.get('analysis'),
                'dry_run': v.get('log', {}).get('dry_run', False),
                'error': v.get('log', {}).get('error'),
            }
        scene_list.append(item)

    return {
        'name': video_dir.name,
        'player': read_json(video_dir / 'player.json'),
        'coaching': read_json(video_dir / 'coaching.json'),
        'info': meta['video'],
        'run_at': meta['run_at'],
        'timings': meta.get('timings', {}),
        'sync': sync,
        'kills': [{'game_sec': k['game_sec'], 'team': k['team'], 'kills': k['kills'],
                   'score_after': k['score_after']} for k in kills if k.get('game_sec') is not None],
        'series': series,
        'scenes': scene_list,
        'top_k': meta.get('settings', {}).get('top_k', 10),
    }


def main():
    videos = []
    for video_dir in sorted(p for p in RESULT_DIR.iterdir() if p.is_dir() and p.name != 'site'):
        payload = video_payload(video_dir)
        if payload:
            videos.append(payload)
            print(f'  {video_dir.name}: {len(payload["scenes"])} scenes, {len(payload["kills"])} kill events')
    if not videos:
        raise SystemExit('No analyzed videos under results/. Run main.py first.')

    data = json.dumps({'videos': videos}, ensure_ascii=False, separators=(',', ':'))
    html = TEMPLATE.read_text(encoding='utf-8').replace('/*__DATA__*/null', data)
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out = SITE_DIR / 'index.html'
    out.write_text(html, encoding='utf-8')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
