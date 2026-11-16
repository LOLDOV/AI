import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from evaluation.sync_eval import compare_kills
from qwen.event_score import extract_events, make_groups, serialize_groups
from vlm.coach import build_coaching
from vod.events import detect_kills
from vod.frames import export_scene_frames
from vod.player import detect_player
from vod.scan import SAMPLE_FPS, scan_video
from vod.scenes import build_scenes
from vod.sync import build_sync
from vod.video import probe

# VOD 한 경기 분석 파이프라인
#   스캔(HUD OCR·시각 신호) -> 플레이어 팀 판별 -> 시간 동기화 -> 킬 검출 -> 장면 후보 -> 대표 프레임
#   -> (선택) VLM 분석 -> 코칭 리포트
# 결과는 results/<영상이름>/ 아래에 단계별 JSON으로 남긴다 (재현 가능한 평가 기록).
#
# 예) python main.py data/vod/LOL_002.mp4
#     python main.py data/vod/LOL_002.mp4 --timeline data/riot/timelines/KR_123.json --vlm

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / 'results'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def load_timeline_groups(path):
    with open(path, encoding='utf-8') as f:
        timeline = json.load(f)
    events = extract_events(timeline)
    return timeline, serialize_groups(make_groups(events), timeline)


def run(video_path, timeline_path=None, use_vlm=False, dry_run=False, rescan=False, top_k=10,
        out_dir=None, progress=None):
    """progress: optional callable(stage, fraction 0~1) for the web app."""
    video_path = Path(video_path)
    out_dir = Path(out_dir) if out_dir else RESULT_DIR / video_path.stem
    report = progress or (lambda stage, fraction: None)
    timings = {}
    started = time.monotonic()

    info = probe(video_path)
    print(f'[{video_path.name}] {info["width"]}x{info["height"]} {info["fps"]}fps {info["duration_sec"]}s')

    scan_path = out_dir / 'scan.json'
    if scan_path.is_file() and not rescan:
        with scan_path.open(encoding='utf-8') as f:
            scan = json.load(f)
        print('  scan: cached')
    else:
        t = time.monotonic()
        scan = scan_video(video_path, progress=lambda f: report('scan', 0.75 * f))
        timings['scan_sec'] = round(time.monotonic() - t, 1)
        save_json(scan_path, scan)
        print(f'  scan: {len(scan["rows"])} samples in {timings["scan_sec"]}s')
    if scan['clock_box'] is None:
        raise ValueError('게임 시계를 찾지 못했습니다. 리그 오브 레전드 리플레이 화면인지 확인하세요.')

    report('player', 0.76)
    player = detect_player(video_path, info['duration_sec'])
    save_json(out_dir / 'player.json', player)
    print(f'  player: team={player["team"]} slot={player["slot"]} confidence={player["confidence"]}')

    sync = build_sync(scan['rows'])
    save_json(out_dir / 'sync.json', sync)
    for seg in sync['segments']:
        print(f'  sync: video {seg["video_start"]:.0f}-{seg["video_end"]:.0f}s '
              f'offset={seg["offset"]}s residual(max)={seg["residual_max"]}s')

    kills = detect_kills(scan['rows'], sync)
    save_json(out_dir / 'kills.json', kills)
    print(f'  kills from HUD score: {len(kills)}')

    timeline_groups = None
    if timeline_path:
        timeline, timeline_groups = load_timeline_groups(timeline_path)
        sync_check = compare_kills(kills, timeline)
        save_json(out_dir / 'sync_check.json', sync_check)
        print(f'  riot timeline: {len(timeline_groups)} groups, '
              f'kill time error median={sync_check["abs_error_median"]}s max={sync_check["abs_error_max"]}s')

    report('scenes', 0.8)
    scenes = build_scenes(scan, sync, kills, timeline_groups, info['duration_sec'], top_k)
    print(f'  scenes: {scenes["scene_count"]} (event-free {scenes["event_free_count"]})')

    # 상위 장면 + 이벤트 없는 장면(점수가 낮아도 누락률 비교를 위해 항상 포함)
    selected = [s for s in scenes['all_scenes'] if s['rank'] <= top_k or s['event_free']]
    t = time.monotonic()
    for i, scene in enumerate(selected):
        report('frames', 0.8 + 0.08 * i / len(selected))
        scene['frames'] = export_scene_frames(video_path, scene, sync, out_dir / 'frames')
    timings['frames_sec'] = round(time.monotonic() - t, 1)
    save_json(out_dir / 'scenes.json', scenes)
    for s in selected:
        print(f'    #{s["rank"]} game {s["game_time"]} score={s["score"]} {s["sources"]}')

    if use_vlm or dry_run:
        from vlm.scene_analyzer import analyze_scene, client_from_env
        client, model = (None, None) if dry_run else client_from_env()
        vlm_results = []
        t = time.monotonic()
        for i, scene in enumerate(selected):
            report('vlm', 0.88 + 0.1 * i / len(selected))
            try:
                result = analyze_scene(scene, scene['frames'], client, model, dry_run=dry_run, player=player)
            except Exception as exc:
                result = {'analysis': None, 'log': {'rank': scene['rank'], 'error': f'{type(exc).__name__}: {exc}'}}
            vlm_results.append({'rank': scene['rank'], 'game_time': scene['game_time'], **result})
            status = 'dry-run' if dry_run else ('ok' if result['analysis'] else 'failed')
            print(f'    vlm #{scene["rank"]}: {status}')
        timings['vlm_sec'] = round(time.monotonic() - t, 1)
        save_json(out_dir / 'vlm.json', vlm_results)

    timings['total_sec'] = round(time.monotonic() - started, 1)
    save_json(out_dir / 'run_meta.json', {
        'video': info,
        'timeline': str(timeline_path) if timeline_path else None,
        'run_at': datetime.now().astimezone().isoformat(),
        'settings': {'sample_fps': scan.get('sample_fps', SAMPLE_FPS), 'top_k': top_k,
                     'vlm': use_vlm, 'dry_run': dry_run},
        'clock_box': scan['clock_box'],
        'timings': timings,
    })

    report('coach', 0.98)
    coaching = build_coaching(out_dir, use_llm=use_vlm and not dry_run)
    save_json(out_dir / 'coaching.json', coaching)
    print(f'  coaching: {coaching["source"]}, {len(coaching["improvements"])} improvement points')
    print(f'  done in {timings["total_sec"]}s -> {out_dir}')
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='LoL VOD scene detection and coaching pipeline')
    parser.add_argument('videos', nargs='+', help='VOD file(s)')
    parser.add_argument('--timeline', help='Riot match timeline JSON for the (single) video')
    parser.add_argument('--vlm', action='store_true', help='call Qwen VLM for top scenes')
    parser.add_argument('--dry-run', action='store_true', help='build VLM prompts without calling the API')
    parser.add_argument('--rescan', action='store_true', help='ignore cached scan.json')
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    if args.timeline and len(args.videos) > 1:
        parser.error('--timeline can only be used with a single video')

    for video in args.videos:
        run(video, args.timeline, args.vlm, args.dry_run, args.rescan, args.top_k)


if __name__ == '__main__':
    main()
