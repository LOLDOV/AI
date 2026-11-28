"""Fair death-scene benchmark for full / no_verify / vlm_only.

Only the evaluation runner is added; deployed service and existing link.run are untouched.
The three conditions get IDENTICAL 16-second death windows and 8 identical MP4 frames.
Riot/clock are used to locate these test windows; vlm_only does NOT receive the Riot
match, death record, RAG docs or OCR clock as model inputs.  This is a COACHING
ablation, not a VLM-only death-localization benchmark.

Commands:
    python -m experiments.three_conditions.runner run --table gt_videos.csv --run-tag pilot01
    python -m experiments.three_conditions.runner run --table gt_videos.csv --run-tag pilot01 --execute
    python -m experiments.three_conditions.runner sheet --run-tag pilot01
    python -m experiments.three_conditions.runner score --run-tag pilot01
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from pipeline.core import ROOT, dump, load, safe_id, video_meta, file_fingerprint
from link import coach, riot_facts, timesync, verify
from link.patch_docs import PatchDocs
from link.run import BEFORE, AFTER, FRAMES, model_name

CONDITIONS = ("full", "no_verify", "vlm_only")
OUT_BASE = ROOT / "results" / "ablation_3"
LABELS = ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "UNKNOWN", "NOT_FACT")
RUBRIC = ("situation_0_2", "cause_0_2", "actionability_0_2", "evidence_0_2")
MATCH_ID_RE = re.compile(r"^[A-Za-z0-9]+_\d+$")
REDACT = re.compile(r"RGAPI-[A-Za-z0-9-]+|AIza[A-Za-z0-9_-]{20,}")


def _stable(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hash() -> dict:
    names = ("experiments/three_conditions/runner.py", "link/coach.py", "link/verify.py",
             "link/riot_facts.py", "link/timesync.py", "vod/sync.py", "RAG/coaching.py", "link/patch_docs.py")
    return {n: _file_hash(ROOT / n) for n in names if (ROOT / n).is_file()}


def _blank(raw) -> bool:
    return str(raw or "").strip() == ""


def read_table(path: Path, only: set[str] | None = None) -> list[dict]:
    """Read independent VIDEO METADATA (not correctness labels)."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV 파일 없음: {path}")
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("빈 gt_videos.csv")
        required = {"video_id", "match_id", "champion"}
        absent = required - set(reader.fieldnames)
        if absent:
            raise ValueError(f"gt_videos.csv 필수 열 누락: {', '.join(sorted(absent))}")
        for line, original in enumerate(reader, 2):
            if not original or not any(not _blank(v) for v in original.values()):
                continue
            row = {str(k): str(v or "").strip() for k, v in original.items() if k is not None}
            video_id = safe_id(row["video_id"])
            if only is not None and video_id not in only:
                continue
            if video_id in seen:
                raise ValueError(f"{line}행 video_id 중복: {video_id}")
            seen.add(video_id)
            if not MATCH_ID_RE.fullmatch(row["match_id"]):
                raise ValueError(f"{line}행 match_id 형식 오류: {row['match_id']!r}")
            if not row["champion"]:
                raise ValueError(f"{line}행 champion 누락")
            explicit = row.get("video_path") or f"videos/{video_id}.mp4"
            video = Path(explicit)
            if not video.is_absolute():
                video = ROOT / video
            video = video.resolve()
            # Metadata rows are not allowed to reference a video of a different name accidentally.
            if not video.name.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
                raise ValueError(f"{line}행 비디오 파일 확장자 확인: {video}")
            offset_text = row.get("sync_offset_sec", "")
            offset = None
            if offset_text:
                offset = float(offset_text)
                if not math.isfinite(offset) or abs(offset) > 86400:
                    raise ValueError(f"{line}행 부적절한 sync_offset_sec")
            rows.append({"video_id": video_id, "video": video, "match_id": row["match_id"].upper(),
                         "champion": row["champion"], "riot_id": row.get("riot_id") or None,
                         "puuid": row.get("puuid") or None, "sync_offset_sec": offset})
    if only:
        missing = only - seen
        if missing:
            raise ValueError(f"gt_videos.csv에 요청 영상 없음: {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError("검증할 영상 행이 0개입니다")
    return rows


def _match_identity(row: dict, match: dict, timeline: dict):
    if match.get("metadata", {}).get("matchId", "").upper() != row["match_id"]:
        raise ValueError(f"Match ID 불일치: {row['video_id']}")
    if timeline.get("metadata", {}).get("matchId", "").upper() != row["match_id"]:
        raise ValueError(f"Timeline Match ID 불일치: {row['video_id']}")
    target = riot_facts.find_player(match, puuid=row["puuid"], riot_id=row["riot_id"],
                                    champion=row["champion"])
    if row['puuid'] and target.get('puuid') != row['puuid']:
        raise ValueError(f"{row['video_id']}: PUUID와 매치 참가자가 불일치")
    if row['riot_id']:
        norm = lambda v: str(v or '').replace(' ', '').casefold()
        if norm(target.get('riot_id')) != norm(row['riot_id']):
            raise ValueError(f"{row['video_id']}: Riot ID와 매치 참가자가 불일치")
    if str(target["champion"]).casefold() != row["champion"].casefold():
        raise ValueError(f"{row['video_id']}의 플레이어 챔피언 불일치: {target['champion']} vs {row['champion']}")
    return target


def _get_sync(row: dict, folder: Path, *, rescan: bool, allow_manual: bool) -> tuple[dict, dict]:
    """OCR from THIS video, or an explicitly flagged non-OCR fallback."""
    if row["sync_offset_sec"] is not None:
        if not allow_manual:
            raise ValueError(f"{row['video_id']}: 수동 offset은 --allow-manual-sync 없이 사용할 수 없습니다")
        off = row["sync_offset_sec"]
        return {"method": "manual_non_ocr", "offset": off}, {
            "method": "manual_non_ocr", "offset_seconds": off, "elapsed_sec": 0.0,
            "clock_read_rate": None, "segments": [], "warning": "NOT an OCR evaluation"}
    cache = folder / "hud_scan.json"
    stamp_path = folder / "hud_scan_provenance.json"
    sig = file_fingerprint(row["video"])
    if cache.exists() and not rescan:
        if not stamp_path.exists() or load(stamp_path).get("video_fingerprint") != sig:
            raise RuntimeError(f"{row['video_id']}: 영상 provenance 없는 OCR 캐시. --rescan으로 다시 생성하세요")
    started = time.monotonic()
    sync, _ = timesync.ocr_sync(row["video"], cache, rescan=rescan)
    seconds = round(time.monotonic() - started, 3)
    dump(stamp_path, {"video_fingerprint": sig, "video_name": row['video'].name})
    return {"method": "ocr", "sync": sync}, {
        "method": "ocr", "offset_seconds": None, "clock_read_rate": sync.get("clock_read_rate"),
        "segments": sync.get("segments", []), "elapsed_sec_current_invocation": seconds,
        "note": "cached OCR reuse possible; this is not an uncached end-to-end latency"}


def _scene_for_death(d: dict, sync: dict, duration: float) -> dict:
    if sync["method"] == "ocr":
        anchor = timesync.to_video(sync["sync"], d["game_sec"], duration)
    else:
        anchor = d["game_sec"] + sync["offset"]
        if not (0 <= anchor <= duration):
            anchor = None
    sc = {"scene_id": d['death_id'], "type": "death", "death": d,
          "death_game_sec": d['game_sec'], "anchor_sec": anchor}
    if anchor is None:
        sc['skipped'] = 'time_not_covered'
    else:
        sc.update(start_sec=round(max(0.0, anchor - BEFORE), 3),
                  end_sec=round(min(duration, anchor + AFTER), 3))
    return sc


def _frame_manifest(frames: list[dict]) -> list[dict]:
    return [{"id": f['id'], "video_sec": f['video_sec'], "sha256": _file_hash(Path(f['file']))}
            for f in frames]


def _gen(model: str, champion: str, sc: dict, frames: list, records: dict, docs: dict,
         dest: Path, execute: bool, sources: dict):
    """One generation per unique prompt; cache with all relevant input hashes."""
    prompt = coach.build_prompt(champion, sc, frames, records, docs)
    stamp = {"model": model, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
             "frame_manifest": _frame_manifest(frames), "source_sha256": sources,
             "temperature": 0.0, "response_mime_type": "application/json"}
    if dest.is_file():
        cached = load(dest)
        if cached.get("stamp") != stamp:
            raise RuntimeError(f"캐시 입력 변경: {dest}. 새 --run-tag 사용 또는 기존 결과 별도 백업 필요")
        if not isinstance(cached.get('coaching'), dict):
            raise RuntimeError(f"캐시 손상: {dest}")
        return cached, False
    if not execute:
        return None, False
    parsed, reply = coach.generate(model, champion, sc, frames, records, docs)
    result = {"stamp": stamp, "coaching": parsed, "latency_sec": reply.get('latency_sec'),
              "token_usage": reply.get('token_usage') or {}, "raw_text": reply.get('raw_text'),
              "generated_at_epoch": time.time()}
    dump(dest, result)
    return result, True


def _assemble(condition: str, generated: dict, checked: dict | None, sc: dict, frames: list,
              records: dict, docs: dict, model: str, sync_meta: dict, new_request: bool) -> dict:
    if condition == "full":
        if checked is None:
            raise ValueError("Full requires claim checking")
        shown = verify.apply(generated['coaching'], checked)
    else:
        # Crucial: no_verify NEVER calls verify(); it returns raw claims unchanged.
        shown = {**generated['coaching'], 'held_claims': []}
    return {"condition": condition, "scene_id": sc['scene_id'],
            "start_sec": sc['start_sec'], "end_sec": sc['end_sec'],
            "death_game_sec": sc['death_game_sec'], "model": model,
            "sync_method": sync_meta['method'],
            "coaching": shown, "raw_coaching": generated['coaching'],
            "check": ({"applied": True, "counts": checked['counts'],
                       "support_rate_not_truth": checked['support_rate']} if checked is not None
                      else {"applied": False, "counts": None, "support_rate_not_truth": None}),
            "frames": _frame_manifest(frames), "generation_latency_sec": generated.get('latency_sec'),
            "token_usage": generated.get('token_usage'), "new_gemini_request": new_request,
            "record_count_provided": len(records), "patch_doc_count_provided": len(docs),
            "note": "verification is evidence-ID checking, not independent visual truth evaluation"}


def _results_path(out: Path, condition: str, scene_id: str) -> Path:
    return out / condition / f"{scene_id}.json"


def run_one(row: dict, args, *, out_root: Path | None = None) -> dict:
    root = out_root or OUT_BASE
    out = root / safe_id(args.run_tag) / row['video_id']
    out.mkdir(parents=True, exist_ok=True)
    video = row['video']
    if not video.is_file():
        raise FileNotFoundError(f"{row['video_id']}: 영상이 없습니다: {video}")
    meta = video_meta(video)
    print(f"\n[{row['video_id']}] match={row['match_id']} champion={row['champion']}, video={meta['duration_sec']:.1f}s", flush=True)
    match, timeline = riot_facts.load(row['match_id'], fetch=args.fetch)
    player = _match_identity(row, match, timeline)
    player_raw = next(p for p in match['info']['participants'] if p['participantId'] == player['participant_id'])
    deaths = riot_facts.deaths(match, timeline, player['participant_id'])
    if args.limit_deaths is not None:
        deaths = deaths[:args.limit_deaths]
    if not deaths:
        raise ValueError(f"{row['video_id']}: Riot 데스 이벤트가 0개. 해당 영상에 데스가 없다면 death benchmark 제외")
    sync, sync_meta = _get_sync(row, out, rescan=args.rescan, allow_manual=args.allow_manual_sync)
    scenes = [_scene_for_death(d, sync, meta['duration_sec']) for d in deaths]
    if not any('skipped' not in s for s in scenes):
        raise ValueError(f"{row['video_id']}: 모든 사망 이벤트가 MP4 범위 밖. Riot 경기와 VOD 시각을 확인하세요")
    version = riot_facts.game_version(match)
    # RAG documents are created exclusively for Full/No Verify and shared, never VLM-only.
    docs_index = PatchDocs(version)
    model = model_name(args.model)
    sources = _source_hash()
    scene_plan = [{"scene_id": s['scene_id'], "start_sec": s.get('start_sec'),
                   "end_sec": s.get('end_sec'), "skipped": s.get('skipped')}
                  for s in scenes]
    signature = {"match_id": row['match_id'], "champion": player['champion'],
                 "participant_id": player['participant_id'], "game_version": version,
                 "video_partial_fingerprint": file_fingerprint(video),
                 "sync": sync_meta, "model": model, "code_hashes": sources,
                 "window_before_sec": BEFORE, "window_after_sec": AFTER, "frames_per_scene": FRAMES,
                 "requested_death_count": len(deaths), "scene_plan_hash": _stable(scene_plan),
                 "sync_method": sync_meta["method"],
                 "sync_segments_hash": _stable(sync_meta.get("segments", []) or sync_meta.get("offset_seconds"))}
    previous = out / 'experiment_manifest.json'
    if previous.is_file():
        prev = load(previous)
        # allow metadata such as timings to change across resume, but not input provenance.
        keys = ("match_id", "champion", "participant_id", "game_version", "video_partial_fingerprint",
                "model", "code_hashes", "window_before_sec", "window_after_sec", "frames_per_scene",
                "requested_death_count", "scene_plan_hash", "sync_method", "sync_segments_hash")
        if any(prev.get(k) != signature[k] for k in keys):
            raise RuntimeError(f"실험 조건 변경: {out}. 새 --run-tag를 사용하세요")
    else:
        dump(previous, signature)
    dump(out / 'scenes.json', scenes)
    attempted = 0
    failures = []
    requests = 0
    for sc in scenes:
        if sc.get('skipped'):
            print('  [SKIP]', sc['scene_id'], 'sync out of range')
            continue
        scene_path = out / 'frames' / sc['scene_id']
        frames = coach.capture(video, sc, scene_path, FRAMES)
        if len(frames) != FRAMES:
            raise RuntimeError(f"{row['video_id']}/{sc['scene_id']}: {len(frames)}/{FRAMES} frames")
        death = sc['death']
        records = coach.record_lines(death, match, timeline, player, docs_index.local_name)
        docs = coach.doc_lines(docs_index.for_death(player['champion'], death, player_raw))
        # NoVerify uses EXACT same generated dict as Full. No extra model call.
        for kind in ('assisted', 'vlm_only'):
            rec = records if kind == 'assisted' else {}
            knowledge = docs if kind == 'assisted' else {}
            cache = out / 'generation' / sc['scene_id'] / f'{kind}.json'
            try:
                generated, is_new = _gen(model, player['champion'], sc, frames, rec, knowledge,
                                         cache, args.execute, sources)
                if generated is None:
                    print(f"  [PLAN] {sc['scene_id']} {kind} (8 same frames)", flush=True)
                    continue
                requests += int(is_new)
                if kind == 'assisted':
                    started = time.monotonic()
                    checked = verify.verify(generated['coaching'],
                        verify.build_context(frames, records, docs, docs_index, match, death))
                    verify_sec = round(time.monotonic() - started, 4)
                    for condition in ('full', 'no_verify'):
                        item = _assemble(condition, generated, checked if condition == 'full' else None,
                                         sc, frames, records, docs, model, sync_meta, is_new if condition == 'full' else False)
                        item['verification_elapsed_sec'] = verify_sec if condition == 'full' else 0.0
                        dump(_results_path(out, condition, sc['scene_id']), item)
                else:
                    item = _assemble('vlm_only', generated, None, sc, frames, {}, {}, model, sync_meta, is_new)
                    item['verification_elapsed_sec'] = 0.0
                    dump(_results_path(out, 'vlm_only', sc['scene_id']), item)
            except Exception as exc:
                err = REDACT.sub('[REDACTED]', str(exc))[:500]
                dump(out / 'errors' / f"{sc['scene_id']}_{kind}.json", {'type': type(exc).__name__, 'error': err})
                print(f"  [ERROR] {sc['scene_id']} {kind}: {err}", flush=True)
                if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                    raise RuntimeError(f"Gemini quota/rate-limit. Already generated results retained. {err}") from exc
                failures.append(f'{sc["scene_id"]}:{kind}')
        attempted += 1
    summaries = {}
    for cond in CONDITIONS:
        files = sorted((out / cond).glob('death_*.json')) if (out / cond).exists() else []
        rows = [load(p) for p in files]
        n = len(rows)
        all_claims = sum(len(r['raw_coaching']['claims']) for r in rows)
        visible = sum(len(r['coaching']['claims']) for r in rows)
        summaries[cond] = {'scenes': n, 'raw_claims': all_claims, 'displayed_claims': visible,
                           'display_coverage': round(visible / all_claims, 4) if all_claims else None,
                           'gemini_latency_sec_sum_reported': round(sum(r['generation_latency_sec'] or 0 for r in rows), 2),
                           'token_total_reported': sum((r['token_usage'] or {}).get('total_tokens') or 0 for r in rows),
                           'note': 'full/no_verify share ONE generated response. Do NOT sum their token counts as two distinct paid requests.'}
    expected = len([s for s in scenes if 'skipped' not in s])
    for cond in CONDITIONS:
        summaries[cond]['expected_scenes'] = expected
        summaries[cond]['complete'] = summaries[cond]['scenes'] == expected
    dump(out / 'summary.json', {'video_id': row['video_id'], 'match_id': row['match_id'],
                                'model': model, 'conditions': summaries, 'new_gemini_requests_this_invocation': requests,
                                'errors': failures, 'sync': sync_meta, 'scenes_prepared': attempted,
                                'caution': 'Human GT is required; this is not accuracy. Full/no_verify share generation. Video-match identity is user supplied, not proven by MP4.'})
    print(f"  [DONE] scenes prepared={attempted}; new Gemini calls={requests}; errors={len(failures)}", flush=True)
    return {'video_id': row['video_id'], 'output': str(out), 'conditions': summaries, 'errors': failures}


def run(args):
    only = {safe_id(x) for x in args.videos.split(',')} if args.videos else None
    rows = read_table(Path(args.table).resolve(), only=only)
    if args.execute:
        print('!!! --execute: Gemini paid API calls enabled. Existing valid generation cache reused. !!!', flush=True)
    else:
        print('DRY RUN: prepares shared death scenes and eight frames; makes NO Gemini calls.', flush=True)
    all_results = []
    for row in rows:
        try:
            all_results.append(run_one(row, args))
        except Exception as exc:
            safe = REDACT.sub('[REDACTED]', str(exc))[:600]
            print(f"[VIDEO ERROR] {row['video_id']}: {safe}", flush=True)
            all_results.append({'video_id': row['video_id'], 'error': safe})
            if not args.continue_on_error or '429' in safe:
                break
    root = OUT_BASE / safe_id(args.run_tag)
    dump(root / 'batch_status.json', {'rows': all_results, 'requested_videos': [r['video_id'] for r in rows],
                                       'execute': bool(args.execute), 'complete': len(all_results) == len(rows)
                                        and all('error' not in r for r in all_results)})
    if any('error' in x for x in all_results) or any(x.get('errors') for x in all_results):
        raise SystemExit(1)
    return root


def _collected_results(tag: str) -> list[dict]:
    root = OUT_BASE / safe_id(tag)
    items = []
    for video_dir in sorted(root.iterdir()) if root.exists() else []:
        if not video_dir.is_dir():
            continue
        for condition in CONDITIONS:
            for path in sorted((video_dir / condition).glob('death_*.json')):
                data = load(path)
                if data.get('condition') != condition:
                    raise RuntimeError(f'조건 혼입: {path}')
                items.append({'video_id': video_dir.name, 'condition': condition, **data})
    return items


def _claim_token(claim: dict) -> str:
    """Stable content ID: Gemini may repeat the same claim.id for different facts."""
    return _stable({k: claim.get(k) for k in ('id', 'kind', 'text', 'refs')})


def _shown_claim_indices(raw: list[dict], visible: list[dict]) -> list[int]:
    from collections import Counter
    counts = Counter(_claim_token(c) for c in visible)
    found = []
    for idx, claim in enumerate(raw, 1):
        token = _claim_token(claim)
        if counts[token] > 0:
            counts[token] -= 1
            found.append(idx)
    return found


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def sheet(args):
    import random
    tag = safe_id(args.run_tag)
    rows = _collected_results(tag)
    if not rows:
        raise ValueError('분석 결과 없음. 먼저 run --execute를 완료하세요')
    if not args.allow_partial:
        run_root = OUT_BASE / tag
        for video_dir in sorted(run_root.iterdir()):
            if not video_dir.is_dir() or video_dir.name == 'human_review':
                continue
            summ = video_dir / 'summary.json'
            if not summ.is_file():
                raise RuntimeError(f'{video_dir}: summary.json 없음')
            complete = load(summ).get('conditions', {})
            if not all(complete.get(c, {}).get('complete') for c in CONDITIONS):
                raise RuntimeError(f'{video_dir.name}: 3조건 결과가 모두 준비되지 않음. --allow-partial은 탐색적 평가에만 사용')
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    root = OUT_BASE / tag / 'human_review'
    root.mkdir(parents=True, exist_ok=True)
    rating = []
    claim_rating = []
    key = {}
    for idx, r in enumerate(rows, 1):
        item_id = f'item_{idx:04d}'
        key[item_id] = {'video_id': r['video_id'], 'scene_id': r['scene_id'], 'condition': r['condition'],
                        'frame_hashes': [x['sha256'] for x in r['frames']],
                        'visible_claim_indices': _shown_claim_indices(r['raw_coaching'].get('claims', []),
                                               r['coaching'].get('claims', [])),
                        'claim_count': len(r['raw_coaching'].get('claims', []))}
        text, _ = coach_text(r['coaching'])
        rating.append({'item_id': item_id, 'video_id': r['video_id'],
                       'start_sec': r['start_sec'], 'end_sec': r['end_sec'], 'coaching_text': text,
                       **{c: '' for c in RUBRIC}, 'memo': ''})
        for n, claim in enumerate(r['raw_coaching']['claims'], 1):
            claim_rating.append({'item_id': item_id, 'claim_no': n,
                                 'claim_text': claim.get('text', ''), 'human_label': '', 'evidence_ref': '', 'memo': ''})
    _write_csv(root / 'coaching_rating.csv', ['item_id', 'video_id', 'start_sec', 'end_sec', 'coaching_text', *RUBRIC, 'memo'], rating)
    _write_csv(root / 'claim_review.csv', ['item_id', 'claim_no', 'claim_text', 'human_label', 'evidence_ref', 'memo'], claim_rating)
    dump(root / 'rating_key_DO_NOT_SHOW_RATERS.json', key)
    print('[SHEET]', len(rating), 'coaching outputs,', len(claim_rating), 'claims ->', root)
    print('IMPORTANT: keep rating_key_DO_NOT_SHOW_RATERS.json hidden from human raters.')
    return root


def coach_text(item: dict):
    out = [f"{item.get('title', '')}", f"상황: {item.get('situation', '')}", f"원인: {item.get('cause', '')}"]
    for c in item.get('claims', []):
        out.append(f"- {c.get('text', '')}")
    for a in item.get('alternatives', []):
        out.append(f"→ {a.get('action', '')} ({a.get('reason', '')})")
    return '\n'.join(out), None


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def score(args):
    root = OUT_BASE / safe_id(args.run_tag) / 'human_review'
    key_path = root / 'rating_key_DO_NOT_SHOW_RATERS.json'
    if not key_path.exists():
        raise ValueError('sheet 실행 후 rating CSV를 작성하세요')
    key = load(key_path)
    rating_path = Path(args.ratings).resolve() if args.ratings else root / 'coaching_rating.csv'
    claim_path = Path(args.claims).resolve() if args.claims else root / 'claim_review.csv'
    coaching_rows = _read_csv(rating_path)
    claim_rows = _read_csv(claim_path)
    coaching_ids = [r.get('item_id') for r in coaching_rows]
    if len(coaching_ids) != len(set(coaching_ids)):
        raise ValueError('코칭 평가표 item_id 중복')
    claim_keys = [(r.get('item_id'), r.get('claim_no')) for r in claim_rows]
    if len(claim_keys) != len(set(claim_keys)):
        raise ValueError('주장 평가표 item_id,claim_no 중복')
    def fresh():
        return {'items_total': 0, 'items_rated': 0, 'rubric_sum': 0,
                                'claims_total': 0, 'claims_rated': 0, 'shown_claims_total': 0, 'shown_claims_rated': 0,
                               'shown_supported': 0, 'shown_contradicted': 0,
                               'shown_unsupported': 0, 'shown_unknown': 0, 'shown_not_fact': 0,
                               'supported': 0,
                               'contradicted': 0, 'unsupported': 0, 'unknown': 0, 'not_fact': 0}
    by_cond: dict[str, dict] = defaultdict(fresh)
    by_video: dict[tuple[str, str], dict] = defaultdict(fresh)
    for data in key.values():
        for box in (by_cond[data['condition']], by_video[(data['video_id'], data['condition'])]):
            box['items_total'] += 1
    for row in coaching_rows:
        key_id = row['item_id']
        if key_id not in key:
            raise ValueError(f'평가표에 알 수 없는 item_id: {key_id}')
        if not any(row.get(c, '').strip() for c in RUBRIC):
            continue
        values = []
        for c in RUBRIC:
            v = row.get(c, '').strip()
            if v not in {'0', '1', '2'}:
                raise ValueError(f'{key_id} {c}에는 0/1/2 모두 채워야 함')
            values.append(int(v))
        for box in (by_cond[key[key_id]['condition']], by_video[(key[key_id]['video_id'], key[key_id]['condition'])]):
            box['items_rated'] += 1
            box['rubric_sum'] += sum(values)
    for row in claim_rows:
        key_id = row['item_id']
        if key_id not in key:
            raise ValueError(f'평가표에 알 수 없는 item_id: {key_id}')
        claim_index = int(row['claim_no']) - 1
        actual_count = key[key_id]['claim_count']
        if claim_index < 0 or claim_index >= actual_count:
            raise ValueError(f"{key_id}: claim_no 범위를 벗어남")
        shown = (claim_index + 1) in key[key_id]['visible_claim_indices']
        label = row.get('human_label', '').strip().upper()
        if label and label not in LABELS:
            raise ValueError(f'{key_id}의 잘못된 라벨: {label}')
        for box in (by_cond[key[key_id]['condition']], by_video[(key[key_id]['video_id'], key[key_id]['condition'])]):
            box['claims_total'] += 1
            if shown:
                box['shown_claims_total'] += 1
            if label:
                box['claims_rated'] += 1
                box[label.lower()] += 1
                if shown:
                    box['shown_claims_rated'] += 1
                    box['shown_' + label.lower()] += 1
    def stats_row(cond: str, x: dict, video_id: str | None = None) -> dict:
        verified = x['supported'] + x['contradicted']
        factual = x['supported'] + x['contradicted'] + x['unsupported'] + x['unknown']
        shown_verified = x['shown_supported'] + x['shown_contradicted']
        shown_factual = x['shown_supported'] + x['shown_contradicted'] + x['shown_unsupported'] + x['shown_unknown']
        row = {'condition': cond, **x,
               'coaching_avg_0_8': round(x['rubric_sum'] / x['items_rated'], 3) if x['items_rated'] else '',
               'fact_accuracy_verified': round(x['supported'] / verified, 4) if verified else '',
               'explicit_error_rate_verified': round(x['contradicted'] / verified, 4) if verified else '',
               'evidence_support_rate_all_factual': round(x['supported'] / factual, 4) if factual else '',
               'displayed_fact_accuracy_verified': round(x['shown_supported'] / shown_verified, 4) if shown_verified else '',
               'displayed_explicit_error_rate_verified': round(x['shown_contradicted'] / shown_verified, 4) if shown_verified else '',
               'displayed_support_rate_all_factual': round(x['shown_supported'] / shown_factual, 4) if shown_factual else '',
               'correct_claims_held_by_verifier': x['supported'] - x['shown_supported'],
               'claim_rating_complete': x['claims_total'] == x['claims_rated'],
               'coaching_rating_complete': x['items_total'] == x['items_rated']}
        return {'video_id': video_id, **row} if video_id is not None else row
    scores = [stats_row(cond, by_cond[cond]) for cond in CONDITIONS]
    per_video_scores = [stats_row(cond, by_video[(vid, cond)], vid)
                        for vid in sorted({x['video_id'] for x in key.values()}) for cond in CONDITIONS]
    names = list(scores[0])
    _write_csv(root / 'human_scores.csv', names, scores)
    dump(root / 'human_scores.json', scores)
    if per_video_scores:
        _write_csv(root / 'human_scores_by_video.csv', list(per_video_scores[0]), per_video_scores)
    print('[SCORED]', root / 'human_scores.csv')
    print('Only fully rated conditions support final comparison. Unrated items are NOT counted as errors.')
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m experiments.three_conditions.runner')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='prepare and optionally execute all 3 fair conditions')
    r.add_argument('--table', default='gt_videos.csv', help='CSV: video_id,match_id,champion[,riot_id,puuid,video_path,sync_offset_sec]')
    r.add_argument('--videos', help='comma separated video IDs subset')
    r.add_argument('--run-tag', default='pilot01')
    r.add_argument('--model', default=None)
    r.add_argument('--fetch', action='store_true', help='fetch missing Riot JSON; may require API key')
    r.add_argument('--rescan', action='store_true', help='rerun OCR from original VOD')
    r.add_argument('--allow-manual-sync', action='store_true', help='explicitly allow CSV manual sync; not OCR evaluation')
    r.add_argument('--limit-deaths', type=int, help='smoke test first N deaths per VOD')
    r.add_argument('--continue-on-error', action='store_true', help='process remaining VOD after errors')
    r.add_argument('--execute', action='store_true', help='call Gemini API; may incur charges')
    sh = sub.add_parser('sheet', help='produce human blind rating sheets')
    sh.add_argument('--run-tag', default='pilot01')
    sh.add_argument('--seed', type=int, default=7)
    sh.add_argument('--allow-partial', action='store_true', help='create an exploratory sheet for incomplete conditions')
    sc = sub.add_parser('score', help='aggregate completed human ratings')
    sc.add_argument('--run-tag', default='pilot01')
    sc.add_argument('--ratings', help='optional filled coaching_rating.csv path')
    sc.add_argument('--claims', help='optional filled claim_review.csv path')
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        if args.limit_deaths is not None and args.limit_deaths <= 0:
            ap.error('--limit-deaths must be positive')
        return run(args)
    if args.cmd == 'sheet':
        return sheet(args)
    if args.cmd == 'score':
        return score(args)

if __name__ == '__main__':
    main()
