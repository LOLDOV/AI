"""Run the three independent OpenCV HUD modules against isolated pipeline outputs."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from .core import ROOT,load
from .evidence import compile_evidence


def analyze(video_id:str,level_zero:bool=False,spells:bool=False,spell_every:float=8.0):
    root=ROOT/'results'/'pipeline'/video_id
    manifest=load(root/'manifest.json')
    base=root/'hud_temporal'
    if not manifest.get('hud_profile'):
        print('[SKIP] HUD profile not provided. Machine HUD analysis unavailable.')
    else:
        for sc in manifest['scenes']:
            folder=base/sc['scene_id']
            if not (folder/'hud_timeline.json').is_file():
                print('[SKIP]',folder);continue
            engines=[('visual_recognizer.py', ['--contact-sheet-step','0'])]
            if level_zero:
                engines.append(('skill_level_zero.py', []))
            engines.append(('cc_context.py',[]))
            for engine,extra in engines:
                cmd=[sys.executable,str(ROOT/'pipeline'/'engines'/engine),
                     '--base',str(base),'--only',sc['scene_id'],*extra]
                print('[HUD]',sc['scene_id'],engine,flush=True)
                subprocess.run(cmd,cwd=str(ROOT),check=True)
    if spells and not manifest.get('hud_profile'):
        print('[SKIP] Spell probe needs a calibrated HUD profile.')
    elif spells:
        source=ROOT/'pipeline'/'spells.py'
        if not source.is_file():
            print('[SKIP] spell probe missing:',source)
        else:
            cmd=[sys.executable,str(source),'--game',video_id,
                 '--scene-root',str(base),
                 '--outdir',str(root/'summoner_spells'),
                 '--hud-profile',str(manifest['hud_profile']),
                 '--video',manifest['video_path'],'--every',str(spell_every)]
            print('[SPELL] optional template matching (may download references).',flush=True)
            subprocess.run(cmd,cwd=str(ROOT),check=True)
    return compile_evidence(manifest)
