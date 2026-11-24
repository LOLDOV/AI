"""One-job-at-a-time local worker, reusing unchanged pipeline and RAG CLI.

This is *not* a production distributed queue. Do not host on the public internet.
"""
from __future__ import annotations
import contextlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pipeline.core import ROOT, dump, load
from RIOT.time_sync import estimate_from_video
from .storage import update_job

WORKERS = ThreadPoolExecutor(max_workers=1, thread_name_prefix='loldov-analysis')
LOG_DIR = ROOT / 'results' / 'service_jobs'
SECRET_RE = re.compile(r'RGAPI-[a-zA-Z0-9-]+')


def _job_env():
    env = os.environ.copy()
    try:
        from dotenv import dotenv_values
        for key, value in dotenv_values(ROOT / '.env').items():
            if key and value is not None:
                env[key] = value
    except ImportError:
        pass
    return env


def _call(job_id: str, args: list[str], log, stage: str):
    update_job(job_id, status='running', stage=stage)
    print(f'[{stage}] starting', file=log, flush=True)
    proc = subprocess.Popen([sys.executable, '-m', *args], cwd=str(ROOT), env=_job_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', bufsize=1)
    recent = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = SECRET_RE.sub('[REDACTED_RIOT_KEY]', line)
        log.write(line)
        log.flush()
        recent.append(line.strip())
        if len(recent) > 12:
            recent.pop(0)
    code = proc.wait()
    if code:
        raise RuntimeError(f'{stage} 단계 실패(exit={code}): ' + ' | '.join(recent)[-1200:])


def _hud_ocr_anchor(video: Path, discovery_id: str) -> dict:
    """Clock offset from vod/ RapidOCR scan; cached next to the discovery result."""
    from vod.sync import build_sync
    cache = ROOT / 'results' / 'discovery' / discovery_id / 'hud_scan.json'
    if cache.is_file():
        scan = load(cache)
    else:
        from vod.scan import scan_video
        scan = scan_video(video, log_every=0)
        dump(cache, scan)
    if scan.get('clock_box') is None:
        raise RuntimeError('game clock not found on screen')
    sync = build_sync(scan['rows'])
    if not sync['segments']:
        raise RuntimeError('no consistent clock segment')
    # RAG build takes one offset; use the longest segment if the replay was paused or sped up.
    seg = max(sync['segments'], key=lambda s: s['points'])
    return {'video_sec': seg['video_start'], 'game_sec': seg['game_start'],
            'segments': sync['segments'], 'clock_read_rate': sync['clock_read_rate']}


def _death_coaching(job_id: str, video: Path, match_id: str, puuid: str, discovery_id: str, log):
    """Death-scene coaching (link/). Failure here keeps the main report.
    Output: results/link/<discovery_id>/web/full/results.json"""
    update_job(job_id, stage='death_coaching')
    print('[death_coaching] starting', file=log, flush=True)
    try:
        from link.run import main as link_main
        with contextlib.redirect_stdout(log):
            link_main(['run', '--video', str(video), '--video-id', discovery_id, '--match-id', match_id,
                       '--puuid', puuid, '--conditions', 'full', '--run-tag', 'web',
                       '--scan-cache', str(ROOT / 'results' / 'discovery' / discovery_id / 'hud_scan.json'),
                       '--fetch', '--execute'])
    except (Exception, SystemExit) as exc:
        print('[death_coaching] failed, main report kept:', SECRET_RE.sub('[REDACTED_RIOT_KEY]', str(exc))[:500],
              file=log, flush=True)


def run_analysis(job_id: str, video: Path, match_id: str, puuid: str, champion: str,
                 role: str, discovery_id: str):
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    log_path = LOG_DIR / f'{job_id}.log'
    try:
        with log_path.open('a', encoding='utf-8') as log:
            print('One selected Riot match is user-provided; identity is not proven by video alone.',file=log)
            _call(job_id, ['pipeline.cli','discover','--video',str(video),
                           '--discovery-id',discovery_id,'--champion',champion,
                           '--role',role,'--execute'], log, 'discover')
            result_path = ROOT / 'results' / 'discovery' / discovery_id / 'discovery.json'
            if not result_path.is_file():
                raise RuntimeError('v4 discovery.json이 생성되지 않았습니다. 로그를 확인하세요')
            discovery = load(result_path)
            if discovery.get('status') == 'complete_no_candidates':
                update_job(job_id,status='complete_no_scenes',stage='done',sync_status='not_applicable')
                return
            if discovery.get('status') != 'complete':
                raise RuntimeError(f'discover 미완료: {discovery.get("status")}')
            update_job(job_id,stage='time_sync')
            cmds = ['RAG.cli','build','--discovery-id',discovery_id,
                    '--match-id',match_id,'--puuid',puuid,
                    '--confirm-same-match','--fetch']
            try:
                anchor = _hud_ocr_anchor(video, discovery_id)
            except Exception as exc:
                anchor = None
                print('[TIME_SYNC] RapidOCR unavailable, falling back to Tesseract:', exc, file=log, flush=True)
            if anchor:
                print('[TIME_SYNC] RapidOCR', anchor, file=log, flush=True)
                cmds += ['--sync-video',str(anchor['video_sec']),
                         '--sync-game',str(anchor['game_sec']), '--sync-source','ocr_clock']
                sync_state = 'ocr_candidate_estimate_not_independently_verified'
            else:
                sync = estimate_from_video(video)
                print('[TIME_SYNC]', sync, file=log,flush=True)
                # Only attach timeline events if repeated clock observations agree.
                if sync.get('status') == 'ocr_candidate_not_riot_event_verified':
                    point = sync['samples'][0]
                    cmds += ['--sync-video',str(point['video_sec']),
                             '--sync-game',str(point['game_sec']), '--sync-source','ocr_clock']
                    sync_state = 'ocr_candidate_estimate_not_independently_verified'
                else:
                    sync_state = 'unavailable_no_timeline_events'
            update_job(job_id,sync_status=sync_state)
            _call(job_id,cmds,log,'riot_rag_build')
            # Default final service makes only assisted coaching; A/B is an experiment.
            _call(job_id,['RAG.cli','coach','--discovery-id',discovery_id,'--execute'],log,'coaching')
            result_dir = ROOT / 'results' / 'discovery' / discovery_id / 'riot_rag' / 'coaching' / 'riot_rag_ab01'
            generated = list(result_dir.glob('scene_*/assisted_result.json'))
            expected = len(discovery.get('selected_scenes',[]))
            if not generated or len(generated) != expected:
                raise RuntimeError(f'coaching 결과 불완전: {len(generated)}/{expected}장면. API 오류/할당량을 확인하세요')
            _death_coaching(job_id, video, match_id, puuid, discovery_id, log)
            update_job(job_id,status='complete',stage='done')
    except Exception as exc:
        error = SECRET_RE.sub('[REDACTED_RIOT_KEY]',str(exc))[:1000]
        update_job(job_id,status='error',stage='failed',error=error)
        with log_path.open('a',encoding='utf-8') as log:
            print('[ERROR]',error,file=log)


def submit_analysis(*args):
    return WORKERS.submit(run_analysis,*args)
