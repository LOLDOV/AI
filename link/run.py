"""Death-scene pipeline runner and comparison conditions.

  full      OCR time sync + Riot records + patch docs
  no_rag    patch docs removed
  no_ocr    time sync by VLM clock reading instead of OCR
  vlm_only  no OCR / Riot / docs; scenes picked by pipeline discover, frames only

Examples
  python -m link.run run --video videos/LOL_036.mp4 --match-id KR_123 --riot-id "name#KR1" --fetch
  python -m link.run run ... --conditions full,vlm_only --execute
  python -m link.run hud --video videos/LOL_036.mp4 --times 03:10,07:42 --execute
  python -m link.run sheet --run-tag r1

Without --execute nothing is sent to Gemini; it only prints the deaths and the time sync.
Output: results/link/<video_id>/<run_tag>/
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path

from pipeline.core import ROOT, dump, load, safe_id, video_meta

from . import coach, riot_facts, timesync, vlm_clock
from .patch_docs import PatchDocs
from .timesync import mmss

CONDITIONS = {  # time sync, Riot records, patch docs
    'full': ('ocr', True, True),
    'no_rag': ('ocr', True, False),
    'no_ocr': ('vlm', True, True),
    'vlm_only': (None, False, False),
}
BEFORE, AFTER, FRAMES = 14.0, 2.0, 8
OUT = ROOT / 'results' / 'link'


def model_name(arg=None):
    return arg or os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')


def death_scenes(deaths: list, sync: dict, duration: float) -> list:
    scenes = []
    for d in deaths:
        anchor = timesync.to_video(sync, d['game_sec'], duration)
        scene = {'scene_id': d['death_id'], 'type': 'death', 'death': d}
        if anchor is None:
            scene['skipped'] = 'not_in_video'
        else:
            scene.update(anchor_sec=anchor, start_sec=round(max(0.0, anchor - BEFORE), 2),
                         end_sec=round(min(duration, anchor + AFTER), 2))
        scenes.append(scene)
    return scenes


def discover_scenes(video: Path, video_id: str, champion: str, model: str):
    """Scenes chosen by the VLM alone (pipeline/combat_discovery). Resumable."""
    from pipeline.combat_discovery import discover
    did = safe_id(f'{video_id}_vlmonly')
    folder = ROOT / 'results' / 'discovery' / did
    started = time.monotonic()
    if not (folder / 'discovery.json').is_file():
        discover(video, did, model=model, champion=champion, execute=True)
    if not (folder / 'discovery.json').is_file():
        status = load(folder / 'scan_status.json') if (folder / 'scan_status.json').is_file() else {}
        raise RuntimeError(f"vlm_only discover not finished ({status.get('status')}); run the same command again to resume")
    scenes = [{'scene_id': f'vlm_{i:02d}', 'type': 'vlm_scene', 'death': None, 'anchor_sec': float(s['end_sec']),
               'start_sec': float(s['start_sec']), 'end_sec': float(s['end_sec']), 'vlm_reason': s.get('why')}
              for i, s in enumerate(load(folder / 'discovery.json').get('selected_scenes') or [], 1)]
    return scenes, round(time.monotonic() - started, 2)


def render(coaching: dict, records: dict, docs: dict):
    """Coaching text for raters and the evidence it points to."""
    lines = [f"[{coaching['verdict']}] {coaching['title']}", f"상황: {coaching['situation']}", f"원인: {coaching['cause']}"]
    used = []
    for c in coaching['claims']:
        lines.append(f"- {c['text']} [{', '.join(c['refs']) or '근거 없음'}]")
        used += c['refs']
    for a in coaching['alternatives']:
        lines.append(f"→ {a['action']} ({a['reason']})")
        used += a['refs']
    evidence = []
    for ref in dict.fromkeys(used):
        if ref in records:
            evidence.append(f'{ref}: {records[ref]}')
        elif ref in docs:
            evidence.append(f"{ref}: [{docs[ref]['patch']}] {docs[ref]['text'][:200]}")
    return '\n'.join(lines), '\n'.join(evidence)


def _ocr(video: Path, work: Path, cache: Path | None, rescan: bool):
    """OCR sync plus the first-run scan time (kept so cached runs still report real latency)."""
    cache = cache or work / 'hud_scan.json'
    timing = work / 'scan_time.json'
    fresh = rescan or not cache.is_file()
    started = time.monotonic()
    sync, rows = timesync.ocr_sync(video, cache, rescan)
    if fresh:
        dump(timing, {'scan_sec': round(time.monotonic() - started, 2)})
    sync['elapsed_sec'] = load(timing)['scan_sec'] if timing.is_file() else None
    sync['api_calls'] = 0
    return sync, rows


def run(args):
    video = Path(args.video).resolve()
    vid = safe_id(args.video_id or video.stem)
    work = OUT / vid
    out = work / safe_id(args.run_tag)
    out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f'unknown condition {unknown}; choose from {", ".join(CONDITIONS)}')
    model = model_name(args.model)
    duration = video_meta(video)['duration_sec']

    started = time.monotonic()
    match, timeline = riot_facts.load(args.match_id, fetch=args.fetch)
    player = riot_facts.find_player(match, puuid=args.puuid, riot_id=args.riot_id, champion=args.champion)
    player_raw = next(p for p in match['info']['participants'] if p['participantId'] == player['participant_id'])
    deaths = riot_facts.deaths(match, timeline, player['participant_id'])
    riot_sec = round(time.monotonic() - started, 2)
    started = time.monotonic()
    docs_index = PatchDocs(riot_facts.game_version(match))
    docs_sec = round(time.monotonic() - started, 2)

    syncs, rows = {}, []
    needed = {CONDITIONS[c][0] for c in conditions}
    if 'ocr' in needed:
        syncs['ocr'], rows = _ocr(video, work, Path(args.scan_cache) if args.scan_cache else None, args.rescan)
    if 'vlm' in needed and args.execute:
        syncs['vlm'] = vlm_clock.vlm_sync(video, work, model)

    print(f"[{vid}] {player['champion']} deaths {len(deaths)} / game {riot_facts.game_version(match)} -> docs {docs_index.version}")
    for name, s in syncs.items():
        print(f'  sync {name}: ' + ', '.join(f"video {mmss(g['video_start'])}~ offset {g['offset']:+.1f}s" for g in s['segments']))
    plan = {'video_id': vid, 'video': str(video), 'match_id': args.match_id.upper(), 'champion': player['champion'],
            'game_version': riot_facts.game_version(match), 'ddragon_version': docs_index.version,
            'deaths': len(deaths), 'model': model, 'conditions': conditions,
            'sync': {k: {'segments': v['segments'], 'clock_read_rate': v['clock_read_rate']} for k, v in syncs.items()}}
    dump(out / 'plan.json', plan)
    if not args.execute:
        print('  dry run: add --execute to call Gemini')
        return out

    summary = {}
    for cond in conditions:
        align, use_records, use_docs = CONDITIONS[cond]
        if align == 'vlm' and 'vlm' not in syncs:
            continue
        if align:
            scenes, align_sec = death_scenes(deaths, syncs[align], duration), syncs[align]['elapsed_sec']
        else:
            scenes, align_sec = discover_scenes(video, vid, player['champion'], model)
        results = []
        for sc in scenes:
            if sc.get('skipped'):
                results.append({'condition': cond, 'video_id': vid, 'scene_id': sc['scene_id'],
                                'death_game_time': mmss(sc['death']['game_sec']), 'skipped': sc['skipped']})
                continue
            death = sc['death']
            frames = coach.capture(video, sc, out / 'frames' / f"{sc['scene_id']}_{sc['start_sec']:.0f}", FRAMES)
            prep = time.monotonic()
            records = coach.record_lines(death, match, timeline, player, docs_index.local_name) if use_records and death else {}
            docs = coach.doc_lines(docs_index.for_death(player['champion'], death, player_raw)) if use_docs and death else {}
            prep_sec = round(time.monotonic() - prep, 3)
            cache = out / 'gen' / f"{sc['scene_id']}_{sc['start_sec']:.1f}_{sc['end_sec']:.1f}_{int(bool(records))}{int(bool(docs))}.json"
            if cache.is_file():
                gen = load(cache)
            else:
                coaching, reply = coach.generate(model, player['champion'], sc, frames, records, docs)
                gen = {'coaching': coaching, 'latency_sec': reply['latency_sec'],
                       'token_usage': reply['token_usage'], 'raw_text': reply['raw_text']}
                dump(cache, gen)
            text, evidence = render(gen['coaching'], records, docs)
            results.append({
                'condition': cond, 'video_id': vid, 'scene_id': sc['scene_id'], 'type': sc['type'],
                'death_game_time': mmss(death['game_sec']) if death else None,
                'killer': death['killer'] if death else None,
                'start_sec': sc['start_sec'], 'end_sec': sc['end_sec'], 'start': mmss(sc['start_sec']), 'end': mmss(sc['end_sec']),
                'hud': timesync.hud_at(rows, sc['anchor_sec']) if align == 'ocr' else None,
                'coaching': gen['coaching'], 'coaching_text': text, 'evidence_text': evidence,
                'records': records, 'docs': docs, 'frames': [f['id'] for f in frames],
                'vlm_sec': gen['latency_sec'], 'prep_sec': prep_sec, 'token_usage': gen['token_usage'],
                'vlm_reason': sc.get('vlm_reason')})
        done = [r for r in results if not r.get('skipped')]
        vlm_total = round(sum(r['vlm_sec'] for r in done), 2)
        total = (align_sec or 0) + (riot_sec if use_records else 0) + (docs_sec if use_docs else 0) \
            + vlm_total + sum(r['prep_sec'] for r in done)
        summary[cond] = {'scenes': len(done), 'skipped': len(results) - len(done), 'time_sync': align or 'none',
                         'align_sec': align_sec, 'riot_sec': riot_sec if use_records else 0,
                         'docs_sec': docs_sec if use_docs else 0, 'vlm_sec_total': vlm_total,
                         'total_sec': round(total, 2), 'avg_sec_per_scene': round(total / len(done), 2) if done else None,
                         'gemini_calls': len(done) + (syncs[align].get('api_calls', 0) if align else 0),
                         'tokens_total': sum((r['token_usage'] or {}).get('total_tokens') or 0 for r in done)}
        folder = out / cond
        folder.mkdir(exist_ok=True)
        dump(folder / 'results.json', results)
        _scenes_csv(folder / 'scenes.csv', results)
        print(f"  {cond}: {len(done)} scenes, {summary[cond]['total_sec']} s")
    dump(out / 'summary.json', {'plan': plan, 'conditions': summary,
                                'notes': ['total_sec = time sync + Riot load + docs load + Gemini + prompt prep',
                                          'vlm_only align_sec is the discover time of this run (short when resumed from cache)',
                                          'HUD level is not read by the OCR (clock and team kills only)']})
    return out


def _scenes_csv(path: Path, results: list):
    cols = ['condition', 'video_id', 'scene_id', 'type', 'death_game_time', 'start', 'end', 'start_sec', 'end_sec',
            'hud_clock', 'hud_blue_kills', 'hud_red_kills', 'vlm_sec', 'skipped']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in results:
            hud = r.get('hud') or {}
            w.writerow({**r, 'hud_clock': hud.get('game_clock'), 'hud_blue_kills': hud.get('blue_kills'),
                        'hud_red_kills': hud.get('red_kills')})


def hud(args):
    """OCR vs VLM HUD readings at given video times (module-level check)."""
    video = Path(args.video).resolve()
    work = OUT / safe_id(args.video_id or video.stem)
    times = [vlm_clock.parse_clock(x) if ':' in x else float(x) for x in args.times.split(',')]
    _, rows = _ocr(video, work, Path(args.scan_cache) if args.scan_cache else None, args.rescan)
    out = [{'video_time': mmss(t), 'method': 'ocr', **(timesync.hud_at(rows, t) or {})} for t in times]
    if args.execute:
        readings, reply = vlm_clock.read_hud(video, times, work / 'hud_crops', model_name(args.model))
        out += [{'video_time': mmss(r['video_sec']), 'method': 'vlm', **r, 'latency_sec': reply['latency_sec']}
                for r in readings]
    with open(work / 'hud_reads.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['video_time', 'method', 'game_clock', 'blue_kills', 'red_kills', 'latency_sec'],
                           extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print('[OK]', work / 'hud_reads.csv')


def sheet(args):
    """Blind rating sheet over every video of a run tag. Conditions go only to the key file."""
    items = [r for path in sorted(OUT.glob(f'*/{safe_id(args.run_tag)}/*/results.json'))
             for r in load(path) if not r.get('skipped')]
    random.Random(args.seed).shuffle(items)
    dest = OUT / f'scoring_{safe_id(args.run_tag)}'
    dest.mkdir(parents=True, exist_ok=True)
    key = {}
    cols = ['item_id', 'video_id', 'start', 'end', 'coaching_text', 'evidence_text',
            'correct_1_0', 'supported_by_shown_evidence_1_0', 'memo']
    with open(dest / 'rating_sheet.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(items, 1):
            item = f'item_{i:03d}'
            key[item] = {'condition': r['condition'], 'scene_id': r['scene_id'], 'video_id': r['video_id']}
            w.writerow({'item_id': item, 'video_id': r['video_id'], 'start': r['start'], 'end': r['end'],
                        'coaching_text': r['coaching_text'], 'evidence_text': r['evidence_text'] or '(프레임만)'})
    dump(dest / 'rating_key_DO_NOT_SHOW_RATERS.json', key)
    print(f'[OK] {len(items)} items -> {dest / "rating_sheet.csv"}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m link.run')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--video', required=True)
    r.add_argument('--video-id')
    r.add_argument('--match-id', required=True)
    r.add_argument('--puuid')
    r.add_argument('--riot-id')
    r.add_argument('--champion')
    r.add_argument('--conditions', default=','.join(CONDITIONS))
    r.add_argument('--run-tag', default='r1')
    r.add_argument('--model')
    r.add_argument('--scan-cache', help='reuse an existing vod scan json (e.g. from the web job)')
    r.add_argument('--fetch', action='store_true', help='download missing match JSON with RIOT_API_KEY')
    r.add_argument('--rescan', action='store_true')
    r.add_argument('--execute', action='store_true', help='call Gemini (paid)')
    h = sub.add_parser('hud')
    h.add_argument('--video', required=True)
    h.add_argument('--video-id')
    h.add_argument('--times', required=True, help='video times, e.g. 03:10,07:42')
    h.add_argument('--model')
    h.add_argument('--scan-cache')
    h.add_argument('--rescan', action='store_true')
    h.add_argument('--execute', action='store_true')
    s = sub.add_parser('sheet')
    s.add_argument('--run-tag', default='r1')
    s.add_argument('--seed', type=int, default=7)
    args = ap.parse_args(argv)
    {'run': run, 'hud': hud, 'sheet': sheet}[args.cmd](args)


if __name__ == '__main__':
    main()
