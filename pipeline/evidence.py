"""Build machine-only, per-video auxiliary evidence. No human labels/readback leakage."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import re
from .core import ROOT,load,dump

ALLOWED_MACHINE_STATES = {
    'cooldown_overlay_candidate','timer_overlay_unconfirmed',
    'colored_icon_candidate','dark_icon_unknown',
    'not_learned_level_zero_candidate','unknown'
}


def get_skill_evidence(scene_dir:Path):
    preferred=scene_dir/'hud_state_candidates_with_levels.json'
    path=preferred if preferred.is_file() else scene_dir/'hud_state_candidates.json'
    if not path.is_file():
        return {'status':'not_available','reason':'HUD visual analysis missing'}
    data=load(path)
    if not isinstance(data.get('frames'),list):return {'status':'invalid'}
    counters={sk:Counter() for sk in 'QWER'}
    for fr in data['frames']:
        for sk in 'QWER':
            state=fr.get('abilities',{}).get(sk,{}).get('visual_state_candidate','unknown')
            if state not in ALLOWED_MACHINE_STATES:state='unknown'
            counters[sk][state]+=1
    return {'status':'machine_candidates','samples':len(data['frames']),
        'states':{sk:dict(counters[sk]) for sk in 'QWER'},
        'level_zero_detector_enabled':preferred.is_file(),
        'warning':'Not verified: candidate states do not prove skill was cast, ready or usable. Level-zero detector is HUD/profile-calibrated.'}


def get_cc_evidence(scene_dir:Path):
    path=scene_dir/'hud_cc_context.json'
    if not path.is_file():return {'status':'not_available'}
    data=load(path);events=[]
    for event in data.get('candidate_events',[]):
        # Explicit allowlist prevents passing human-confirmed timestamps/labels to Gemini.
        events.append({'timecode':event.get('timecode'),
           'darkened_slots':list(event.get('darkened_slots',[])),
           'pattern':event.get('candidate_pattern') or event.get('candidate'),
           'status':'visual_CC_candidate_UNCONFIRMED'})
    return {'status':'machine_candidates','events':events,
        'warning':'Candidate count zero does NOT prove CC absence; no CC type/duration confirmed. Human annotations excluded.'}


def get_spells(video_id:str, explicit_path:Path|None=None):
    path=explicit_path or ROOT/'results'/'pipeline'/video_id/'summoner_spells'/'summoner_spell_probe.json'
    if not path.is_file():return {'status':'not_available'}
    raw=load(path)
    if raw.get('video_id')!=video_id:
        return {'status':'ignored_mismatch_video_id','reason':'Spell probe belongs to different video.'}
    slots={}
    for slot in 'DF':
        obj=raw.get('slots',{}).get(slot,{})
        if obj.get('status')!='human_review_required':continue
        # A template vote is not model confidence; do not mark name as confirmed.
        slots[slot]={'name_candidate':obj.get('tentative_name'),
          'template_vote_support_not_accuracy':obj.get('template_vote_support'),
          'name_confirmed':False,'spell_cast_time_confirmed':False,
          'cooldown_confirmed':False}
    return {'status':'visual_name_candidates','slots':slots,
        'warning':'Same spell icon may change due to jungle Smite evolution; votes are correlated, not validated accuracy.'}


def get_rag(role:str,spells:dict,patch='unknown'):
    kb=ROOT/'pipeline'/'knowledge'/'lol_jungle_smite_2026.json'
    if not kb.is_file():return {'status':'not_available','facts':[]}
    data=load(kb)
    if role.lower() not in {'jungle','jg','정글','jungler'} and not any(
       (v.get('name_candidate') or '').strip() in ('강타','Smite') for v in spells.get('slots',{}).values()):
        return {'status':'not_relevant','facts':[]}
    ids={'pet_three_colors','upgrade_icon_changes','spell_cast_not_inferred'}
    if str(patch) in {'26.1','2026-26.1'}:ids.update({'patch_2026_upgrade_rules','blue_pet_effect'})
    selected=[]
    for fact in data.get('facts',[]):
        if fact.get('id') in ids:
            selected.append({'id':fact.get('id'),'rule':fact.get('text_ko'),'source':fact.get('source')})
    return {'status':'retrieved_game_rules','recorded_patch':patch,
        'facts':selected,
        'warning':'Rules are patch-dependent; no player pet color or upgraded Smite stage inferred without video-specific proof. KB facts are not video observations.'}


def compile_evidence(manifest:dict):
    video_id=manifest['video_id']
    root=ROOT/'results'/'pipeline'/video_id
    details=[]
    for sc in manifest['scenes']:
        scene_dir=root/'hud_temporal'/sc['scene_id']
        details.append({'scene_id':sc['scene_id'],
            'time_start_sec':sc['start_sec'],'time_end_sec':sc['end_sec'],
            'skill_visual':get_skill_evidence(scene_dir),
            'cc_visual':get_cc_evidence(scene_dir)})
    spells=get_spells(video_id)
    result={'schema':'machine_only_evidence-v1','video_id':video_id,
        'source':'automatic_only','no_manual_labels_included':True,
        'scenes':details,'spells':spells,
        'health_point_ocr':{'status':'experimental_not_used',
         'reason':'Previous low-resolution Tesseract/Paddle outputs unreliable; no HP numeric assertions sent automatically.'},
        'game_knowledge':get_rag(manifest.get('role','unknown'),spells,manifest.get('patch','unknown')),
        'ground_truth_policy':'Human confirmation used only offline for evaluation, never model input',
        'limitations':['Exact casts, cooldown seconds, CC type, spell availability, objective involvement and match result are unverified.',
                       'Missing candidate is NOT evidence of absence.',
                       'Results depend on HUD calibration and selected-player HUD visibility.']}
    dump(root/'machine_evidence.json',result)
    return result
