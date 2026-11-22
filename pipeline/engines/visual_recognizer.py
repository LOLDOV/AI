"""LoL spectator HUD Q/W/E/R temporal visual-evidence analyzer (local, no OCR/API).

Reads hud_timeline.json + hud_XXX_ability_row.png produced by
OpenCV/hud_temporal_sampler.py; writes *candidates for human review*, never
confirmed skill casts, exact remaining cooldown times, or spell names.

Run from vlm-project root:
    python OpenCV/hud_temporal_recognizer.py --only qiyana_candidate_01_scene_01
    python OpenCV/hud_temporal_recognizer.py --only qiyana_candidate_01_scene_02
    python OpenCV/hud_temporal_recognizer.py --only qiyana_candidate_01_scene_01 --contact-sheet-step 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT / "results" / "hud_temporal" / "LOL_001"
# Calibrated visually on the 900x132 enlarged ability_row crop of 1282x832 spectator HUD.
# Values describe the center of each ability icon and the *upper timer glyph* region.
# Do NOT include lower-level numerals when examining timer glyphs.
REF_W, REF_H = 900, 132
REGIONS = {
    "Q": {"icon": (164, 15, 260, 86), "timer": (183, 15, 236, 72)},
    "W": {"icon": (282, 15, 378, 86), "timer": (298, 15, 350, 72)},
    "E": {"icon": (393, 15, 489, 86), "timer": (410, 15, 472, 72)},
    "R": {"icon": (507, 15, 606, 86), "timer": (525, 15, 580, 72)},
}


def scaled_rect(rect, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = rect
    x1, x2 = round(x1 * w / REF_W), round(x2 * w / REF_W)
    y1, y2 = round(y1 * h / REF_H), round(y2 * h / REF_H)
    return x1, y1, x2, y2


def crop_rect(im, r):
    x1, y1, x2, y2 = scaled_rect(r, im.shape)
    return im[y1:y2, x1:x2]


def white_timer_mask(bgr):
    # White/gray pixels in timer region. Intentionally EXCLUDES level numerals.
    lo = bgr.min(axis=2)
    hi = bgr.max(axis=2)
    return (lo >= 155) & ((hi.astype(np.int16) - lo.astype(np.int16)) < 48)


def frame_features(path):
    im = cv2.imread(str(path))
    if im is None or im.ndim != 3:
        raise ValueError(f"Unable to open HUD image: {path}")
    # Guardrail: shape must match the crop aspect ratio before applying these boxes.
    if abs(im.shape[1] / im.shape[0] - REF_W / REF_H) > 0.12:
        raise ValueError(f"Unexpected HUD crop aspect ratio: {path} {im.shape}")
    feats = {}
    for key, rects in REGIONS.items():
        icon = crop_rect(im, rects['icon'])
        timer = crop_rect(im, rects['timer'])
        hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
        color = np.mean((hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 80))
        mask = white_timer_mask(timer)
        feats[key] = {"color_fraction": round(float(color), 4),
                      "timer_white_fraction": round(float(mask.mean()), 4),
                      "_timer_mask": mask}
    return im, feats


def pair_change(mask1, mask2):
    if mask1 is None or mask2 is None:
        return None
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2.astype(np.uint8), (mask1.shape[1], mask1.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    return float(np.mean(mask1 != mask2))


def interpret(entry, neighbor_change, timer_thresh, color_thresh, change_thresh):
    col = entry['color_fraction']
    whit = entry['timer_white_fraction']
    # Colorfulness is a clue, not proof of ready state.
    if col >= color_thresh:
        return 'colored_icon_candidate', 'Colorful skill icon; readiness still needs validation'
    # Presence of bright changing glyphs in CENTER (not the small level numeral below).
    if whit >= timer_thresh and neighbor_change is not None and neighbor_change >= change_thresh:
        return 'cooldown_overlay_candidate', 'Bright central overlay changes over time; likely countdown'
    if whit >= timer_thresh:
        return 'timer_overlay_unconfirmed', 'Bright central glyph, without strong adjacent change'
    if col < 0.06 and whit < 0.015:
        return 'dark_icon_unknown', 'Dark icon: unlearned, unavailable, or obscured; cannot decide'
    return 'unknown', 'Visual evidence insufficient'


def save_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(folder, records, step=2, per_sheet=12):
    # 900x132 full-size ability rows; contact sheets are for visual review, not ML training.
    thumbs = records[::step]
    if records and thumbs[-1]['sample_index'] != records[-1]['sample_index']:
        thumbs.append(records[-1])
    for chunk_num, i in enumerate(range(0, len(thumbs), per_sheet), 1):
        chunk = thumbs[i:i + per_sheet]
        cell_w, cell_h = 935, 171
        canvas = np.full((math.ceil(len(chunk)/2)*cell_h, 2*cell_w, 3), 35, dtype=np.uint8)
        for j, entry in enumerate(chunk):
            row, col = j // 2, j % 2
            x0, y0 = col*cell_w+16, row*cell_h+10
            image_path = folder / entry['ability_row_image']
            im = cv2.imread(str(image_path))
            if im is None:
                continue
            display = cv2.resize(im, (900,132), interpolation=cv2.INTER_LINEAR)
            canvas[y0+24:y0+24+132, x0:x0+900] = display
            subtitle = f"{entry['source_timecode']} | sample {entry['sample_index']:03d}"
            cv2.putText(canvas, subtitle, (x0,y0+16),cv2.FONT_HERSHEY_SIMPLEX,.52,(235,235,235),1,cv2.LINE_AA)
        cv2.imwrite(str(folder / f"hud_review_contact_sheet_{chunk_num:02d}.jpg"),canvas)


def analyze_scene(folder, timer_thresh, color_thresh, change_thresh, contact_sheet_step):
    timeline_file = folder / 'hud_timeline.json'
    if not timeline_file.exists():
        raise FileNotFoundError(f"No hud_timeline.json: {timeline_file}")
    data = json.loads(timeline_file.read_text(encoding='utf-8-sig'))
    raw = data.get('frames',[])
    if not raw:
        raise ValueError('HUD timeline has no frames')
    # Never rely on filenames for time order.
    raw = sorted(raw,key=lambda x: float(x['source_timestamp_sec']))
    imgs, features, entries = [], [], []
    for row in raw:
        rel = str(row['files']['ability_row'])
        if Path(rel).name != rel or '..' in rel:
            raise ValueError(f'Unsafe ability_row path: {rel}')
        im, feat = frame_features(folder/rel)
        imgs.append(im)
        features.append(feat)
        entries.append(row)
    all_records, changes, perframe = [], [], []
    previous_states = {}
    for i,(row, feat) in enumerate(zip(entries,features)):
        t = float(row['source_timestamp_sec'])
        result = {'sample_index': row.get('sample_index', i),
                  'source_timestamp_sec':t,'source_timecode':row.get('source_timecode'),
                  'game_timecode':row.get('game_timecode'),
                  'ability_row_image': row['files']['ability_row'], 'abilities':{}}
        flat = {k:result[k] for k in ['sample_index','source_timestamp_sec','source_timecode','game_timecode','ability_row_image']}
        for skill in 'QWER':
            ff = feat[skill]
            prevdiff = pair_change(ff['_timer_mask'],features[i-1][skill]['_timer_mask']) if i>0 else None
            nextdiff = pair_change(ff['_timer_mask'],features[i+1][skill]['_timer_mask']) if i+1<len(features) else None
            neigh = max(x for x in [prevdiff,nextdiff] if x is not None) if (prevdiff is not None or nextdiff is not None) else None
            state,why=interpret(ff,neigh,timer_thresh,color_thresh,change_thresh)
            rec={'visual_state_candidate':state,
                 'color_fraction':ff['color_fraction'],
                 'upper_timer_white_fraction':ff['timer_white_fraction'],
                 'neighbor_glyph_change_fraction':round(neigh,4) if neigh is not None else None,
                 'skill_level':None,'exact_cooldown_seconds':None,'cast_confirmed':False,
                 'evidence':why,'verified_by_human':False}
            result['abilities'][skill]=rec
            flat[f'{skill}_visual_candidate']=state
            flat[f'{skill}_color_fraction']=ff['color_fraction']
            flat[f'{skill}_timer_white_fraction']=ff['timer_white_fraction']
            flat[f'{skill}_glyph_change']=rec['neighbor_glyph_change_fraction']
            if skill in previous_states and previous_states[skill]!=state:
                changes.append({'source_timecode':row.get('source_timecode'),
                                'source_timestamp_sec':t,'skill':skill,
                                'old_visual_candidate':previous_states[skill],
                                'new_visual_candidate':state,
                                'evidence':why,'needs_human_review':True})
            previous_states[skill]=state
        all_records.append(result)
        perframe.append(flat)

    output={'version':'hud-temporal-visual-candidates-v1','generated_at':datetime.now().isoformat(timespec='seconds'),
            'scene_id':data.get('scene_id',folder.name),
            'source_timeline':str(timeline_file), 'frame_count':len(all_records),
            'state_is_ground_truth':False,'skill_cast_detection_performed':False,
            'exact_cooldown_ocr_performed':False,
            'allowed_states':['cooldown_overlay_candidate','timer_overlay_unconfirmed','colored_icon_candidate','dark_icon_unknown','unknown'],
            'interpretation_policy':{
                'cooldown_overlay_candidate':'연속 이미지에서 변하는 중앙 숫자/오버레이 후보. 쿨타임이라는 가설이며 정확한 초 수는 측정 안 함',
                'colored_icon_candidate':'아이콘이 컬러로 보임. ready 후보이지 실제 ready 확정 아님',
                'dark_icon_unknown':'미습득과 기타 비활성 상태를 구분하지 않음',
                'important':'원본 hud_checked.json 덮어쓰기 금지; VLM 코칭에 사실로 직접 전달 금지'},
            'frames':all_records,'visual_transitions_for_review':changes}
    (folder/'hud_state_candidates.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    flatcols=['sample_index','source_timestamp_sec','source_timecode','game_timecode','ability_row_image']
    for sk in 'QWER':
        flatcols += [f'{sk}_visual_candidate',f'{sk}_color_fraction',f'{sk}_timer_white_fraction',f'{sk}_glyph_change']
    save_csv(folder/'hud_visual_features.csv',perframe,flatcols)
    save_csv(folder/'hud_changes_review.csv',changes,
             ['source_timecode','source_timestamp_sec','skill','old_visual_candidate','new_visual_candidate','evidence','needs_human_review'])
    # Preserve ALL existing human labels and CC annotations across re-runs.
    # Never overwrite a manually entered ground truth with an empty template.
    label_file = folder / 'hud_manual_validation.csv'
    existing = {}
    if label_file.exists():
        with label_file.open('r', newline='', encoding='utf-8-sig') as f:
            for saved in csv.DictReader(f):
                filename = (saved.get('ability_row_image') or '').strip()
                if filename:
                    existing[filename] = saved
    transition_times = {c['source_timestamp_sec'] for c in changes}
    label_rows=[]
    ordered_files=set()
    for i,row in enumerate(all_records):
        filename = row['ability_row_image']
        selected = (i==0 or i==len(all_records)-1 or i%4==0
                    or row['source_timestamp_sec'] in transition_times)
        if not selected and filename not in existing:
            continue
        old = existing.get(filename,{})
        template={'source_timecode':row['source_timecode'],
                  'ability_row_image':filename}
        for sk in 'QWER':
            template[f'{sk}_human_label']=old.get(f'{sk}_human_label','')
        template['notes']=old.get('notes','')
        template['cc_status']=old.get('cc_status','')
        template['cc_notes']=old.get('cc_notes','')
        label_rows.append(template)
        ordered_files.add(filename)
    # Keep even orphaned manual records instead of silently dropping any evidence.
    for filename, old in existing.items():
        if filename not in ordered_files:
            label_rows.append(old)
    save_csv(label_file,label_rows,
             ['source_timecode','ability_row_image',*[f'{sk}_human_label' for sk in 'QWER'],
              'notes','cc_status','cc_notes'])
    if contact_sheet_step>0:
        make_contact_sheets(folder,all_records,step=contact_sheet_step)
    counts={sk:{} for sk in 'QWER'}
    for row in all_records:
        for sk in 'QWER':
            state=row['abilities'][sk]['visual_state_candidate']
            counts[sk][state]=counts[sk].get(state,0)+1
    return {'scene_id':output['scene_id'],'frames':len(all_records),
            'transition_review_count':len(changes),'counts':counts,
            'output':str(folder/'hud_state_candidates.json')}


def main():
    ap=argparse.ArgumentParser(description='Qiyana 스킬 HUD 연속 변화 후보 분석 (OpenCV / API 없음)')
    ap.add_argument('--base',type=Path,default=DEFAULT_BASE,help='hud_temporal/LOL_001 directory')
    ap.add_argument('--only',default='qiyana_candidate_01_scene_01',help='scene folder name')
    ap.add_argument('--all',action='store_true',help='process all subfolders containing hud_timeline.json')
    ap.add_argument('--timer-threshold',type=float,default=.035)
    ap.add_argument('--color-threshold',type=float,default=.22)
    ap.add_argument('--change-threshold',type=float,default=.02)
    ap.add_argument('--contact-sheet-step',type=int,default=2,help='0 disables contact sheets, 2 is one per 1s')
    args=ap.parse_args()
    base=args.base.resolve()
    dirs=sorted(p for p in base.iterdir() if p.is_dir() and (p/'hud_timeline.json').exists()) if args.all else [base/args.only]
    if not dirs:
        ap.error('No HUD timeline directories found')
    for d in dirs:
        print(json.dumps(analyze_scene(d,args.timer_threshold,args.color_threshold,args.change_threshold,args.contact_sheet_step),ensure_ascii=False,indent=2))
    print('완료: 자동 판정은 검토 후보이며 스킬 시전/정확한 쿨타임/스킬 레벨 인식은 수행하지 않았습니다. API 비용 없음.')

if __name__=='__main__':
    main()
