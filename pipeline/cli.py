"""CLI: prepare -> analyze -> run. Always run from repository root."""
from __future__ import annotations
import argparse
from pathlib import Path
from .core import ROOT,project_path,safe_id,parse_scene,prepare,load
from .analyze import analyze
from .experiment import run
from .combat_discovery import discover, load_discovered_scenes


def main(argv=None):
    ap=argparse.ArgumentParser(prog='python -m pipeline.cli',description='LOLDOV multi-VOD experiment; do not move legacy directories.')
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Build image-identical A/B inputs per video')
    p.add_argument('--video',required=True,help='videos/LOL_001.mp4')
    p.add_argument('--video-id',default=None)
    p.add_argument('--champion',default='unknown')
    p.add_argument('--role',default='unknown')
    p.add_argument('--patch',default='unknown',help='Actual VOD game patch if verified, e.g. 26.1; not current Data Dragon patch')
    p.add_argument('--scene',action='append',default=[],help='scene_id=03:08-03:43.467 OR 03:08-03:43')
    p.add_argument('--auto-scenes',action='store_true',help='LEGACY: HP-drop heuristic; not recommended for combat discovery')
    p.add_argument('--discovery-id',default=None,help='Load ranked unverified VLM scenes from results/discovery/<ID>/discovery.json')
    p.add_argument('--hud-profile',default=None,help='configs/hud_1282x832.json (must match UI)')
    p.add_argument('--frames-per-scene',type=int,default=8)
    p.add_argument('--hud-step',type=float,default=.5)
    p.add_argument('--force',action='store_true',help='Retired for safety; use a fresh --video-id')
    d=sub.add_parser('discover',help='NEW: full-video VLM combat scouting + dense validation')
    d.add_argument('--video',required=True)
    d.add_argument('--discovery-id',required=True)
    d.add_argument('--champion',default='unknown')
    d.add_argument('--role',default='unknown')
    d.add_argument('--model',default=None)
    d.add_argument('--execute',action='store_true',help='Real Gemini API calls: paid, cached, resumable')
    d.add_argument('--window-sec',type=float,default=24.)
    d.add_argument('--sample-sec',type=float,default=4.)
    d.add_argument('--dense-sec',type=float,default=2.)
    d.add_argument('--batch-windows',type=int,default=4)
    d.add_argument('--verify-top',type=int,default=10)
    d.add_argument('--top-k',type=int,default=5)
    d.add_argument('--max-requests',type=int,default=40,
                   help='Initial API request ceiling; retries can add more calls')
    a=sub.add_parser('analyze',help='Run OpenCV QWER/CC candidates in isolated results folder')
    a.add_argument('--video-id',required=True)
    a.add_argument('--enable-calibrated-level-zero',action='store_true',
                   help='R-level-zero template is sample-specific; do not enable on unseen UI without calibration')
    a.add_argument('--with-spells',action='store_true',help='Optional existing D/F probe, may download official icon references')
    a.add_argument('--spell-every',type=float,default=8.)
    sc=sub.add_parser('score',help='Summarize human A/B scoring after both model outputs exist')
    sc.add_argument('--video-id',required=True)
    sc.add_argument('--run-tag',default='ab01')
    inv=sub.add_parser('inventory',help='Read-only folder/code inventory; does not move/delete files')
    r=sub.add_parser('run',help='Prepare A/B prompts; --execute calls Gemini twice')
    r.add_argument('--video-id',required=True)
    r.add_argument('--model',default=None)
    r.add_argument('--run-tag',default='experiment01')
    r.add_argument('--execute',action='store_true',help='CAUTION: real API calls and usage charges (two requests)')
    r.add_argument('--force',action='store_true',help='Retired for safety; use new --run-tag')
    args=ap.parse_args(argv)
    if args.command=='prepare':
        video=project_path(args.video)
        vid=safe_id(args.video_id or video.stem)
        if args.frames_per_scene<2 or args.frames_per_scene>32 or args.hud_step<=0:
            ap.error('frames-per-scene 2~32; hud-step positive')
        scenes=[parse_scene(x,i) for i,x in enumerate(args.scene,1)]
        if int(bool(args.auto_scenes))+int(bool(scenes))+int(bool(args.discovery_id)) != 1:
            ap.error('Choose exactly one of --scene, --auto-scenes (legacy), or --discovery-id')
        if args.discovery_id:
            scenes=load_discovered_scenes(args.discovery_id,video)
        result=prepare(video,vid,args.champion,args.role,scenes,
            project_path(args.hud_profile) if args.hud_profile else None,
            args.frames_per_scene,args.hud_step,args.auto_scenes,args.force,args.patch)
        print('PREPARED:',result)
        print('Next: python -m pipeline.cli analyze --video-id',vid)
    elif args.command=='discover':
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT/'.env')
        except ImportError:pass
        model=args.model or os.getenv('GEMINI_MODEL','gemini-3.6-flash')
        out=discover(project_path(args.video),safe_id(args.discovery_id),model,
            args.champion,args.role,args.execute,args.window_sec,args.sample_sec,
            args.dense_sec,args.batch_windows,args.verify_top,args.top_k,args.max_requests)
        print(('DISCOVERY RUN:' if args.execute else 'OFFLINE SCAN PLAN:'),out)
        status=out/'scan_status.json'
        if status.is_file():print('STATUS:',load(status))
        else:print('No Gemini API calls. Read scan_plan.json before --execute.')
    elif args.command=='analyze':
        result=analyze(safe_id(args.video_id),args.enable_calibrated_level_zero,
                       args.with_spells,args.spell_every)
        print('MACHINE EVIDENCE:',ROOT/'results'/'pipeline'/args.video_id/'machine_evidence.json')
        print('No human labels used. Scenes:',len(result['scenes']))
    elif args.command=='score':
        from .scoring import score
        print('SCORING:',score(args.video_id,args.run_tag))
    elif args.command=='inventory':
        from .maintenance import inventory
        print('FOLDER AUDIT:',inventory())
    elif args.command=='run':
        # .env can define model, CLI explicit override takes priority.
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT/'.env')
        except ImportError:pass
        model=args.model or os.getenv('GEMINI_MODEL','gemini-3.6-flash')
        path=run(safe_id(args.video_id),model,args.run_tag,args.execute,args.force)
        print(('API EXPERIMENT:' if args.execute else 'OFFLINE PREVIEW:'),path)
        print('Review comparison_template.csv and comparison_guide.md; never auto-claim improved accuracy.')

if __name__=='__main__':
    main()
