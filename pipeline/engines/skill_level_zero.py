"""Identify *zero-level* ability glyphs in LoL's 900x132 ability-row crops.

This is a conservative, non-OCR image-template recognizer for the HUD layout
used in LOL_001 (1282x832 replay, 4x crop), not a general-purpose LoL parser.

A verified-looking 0 underneath a DARK icon means the *candidate* state is
not_learned. A dim icon WITHOUT the 0 remains unknown (for example stun).

This script never overwrites hud_state_candidates.json, manual labels, or
previous CC analysis. No Gemini, Riot API, or network calls.

Usage:
  python OpenCV/hud_skill_level_detector.py --only qiyana_candidate_01_scene_01
  python OpenCV/hud_skill_level_detector.py --only qiyana_candidate_01_scene_02

Outputs (in the scene folder):
  hud_state_candidates_with_levels.json
  hud_skill_levels_review.csv
  hud_level_evaluation.json   (only when manually labeled CSV is available)
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT / "results" / "hud_temporal" / "LOL_001"
REF_W, REF_H = 900, 132
# Centers of LOWER level numerals (never inspect the upper cooldown glyph here).
LEVEL_CENTERS = {"Q": 210, "W": 328, "E": 442, "R": 557}
LEVEL_BOX = (44, 34)  # x ±22; y [87,121) in 900x132 crop
LEVEL_TOP = 87

# Binary reference of the numeral '0' under R in the inspected, level-4 Qiyana
# replay HUD. Stored inline so the user need only copy this single Python file.
# It is a tiny glyph mask, NOT an inference that other scenes start at level 0.
ZERO_GLYPH_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACwAAAAiCAAAAAAB5WStAAAAuUlEQVQ4EZXBAWrDQAADMPn/j/ZSQjtfBoNK8YV4qlv8EQ/1EQ9xql/xEIdacYpVL0G9xCFWEbe6xIpRl7jVJVaMIt6KWDGK+CixYlQcGitG418xGqeKEaNxqhgxKg4VI0bFKjFiVKwSI0bFKjFiVKwSI0bFKjFiVKwSI0bFoWLEqDhUjBglRsWKUcRHiRWjiLciVoy6xK0usWIVcSviEKsu8VKXOMShVpziUCMe4lQf8RRPdYs/4gs/bEA3I5bN+V8AAAAASUVORK5CYII="

PRED = {
    "cooldown_overlay_candidate": "cooldown",
    "colored_icon_candidate": "ready",
    "not_learned_level_zero_candidate": "not_learned",
    "dark_icon_unknown": "unknown",
    "timer_overlay_unconfirmed": "unknown",
    "unknown": "unknown",
}
LABELS = {"cooldown", "ready", "not_learned", "unknown"}


def zero_reference():
    raw = np.frombuffer(base64.b64decode(ZERO_GLYPH_PNG_BASE64), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (LEVEL_BOX[1], LEVEL_BOX[0]):
        raise RuntimeError("Embedded zero reference was damaged")
    return img >= 128


def cut_mask(im, skill):
    h, w = im.shape[:2]
    if abs(w / h - REF_W / REF_H) > .12:
        raise ValueError(f"HUD aspect ratio mismatch: {w}x{h}")
    # Rescale to reference crop size; no special fonts/OCR required.
    if (w, h) != (REF_W, REF_H):
        im = cv2.resize(im, (REF_W, REF_H), interpolation=cv2.INTER_AREA)
    x = LEVEL_CENTERS[skill]
    tile = im[LEVEL_TOP:LEVEL_TOP + LEVEL_BOX[1], x - 22:x + 22]
    if tile.shape[:2] != (LEVEL_BOX[1], LEVEL_BOX[0]):
        raise ValueError("Level tile crop failed")
    lo = tile.min(axis=2).astype(np.int16)
    hi = tile.max(axis=2).astype(np.int16)
    # Bright, near-neutral numeric font. Colored skill artwork is suppressed.
    return (lo >= 130) & ((hi - lo) < 68)


def shifted_mask(mask, dx, dy):
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.uint8), transform,
                          (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0) > 0


def matching_score(mask, reference):
    if np.count_nonzero(mask) < 24:
        return 0.0
    best = 0.0
    for dy in range(-2, 3):
        for dx in range(-3, 4):
            shifted = shifted_mask(reference, dx, dy)
            inter = np.count_nonzero(mask & shifted)
            union = np.count_nonzero(mask | shifted)
            best = max(best, inter / union if union else 0.)
    return float(round(best, 4))


def safe_write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_manual(folder, original, revised, threshold):
    label_file = folder / "hud_manual_validation.csv"
    if not label_file.is_file():
        return None
    old_lookup = {fr["ability_row_image"]: fr for fr in original["frames"]}
    new_lookup = {fr["ability_row_image"]: fr for fr in revised["frames"]}
    comparisons, human_unknown, cc_count = [], 0, 0
    seen = set()
    with label_file.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("ability_row_image") or "").strip()
            if not name or name not in new_lookup:
                continue
            if name in seen:
                raise ValueError(f"Duplicate manual label image: {name}")
            seen.add(name)
            cc_status = (row.get("cc_status") or "unknown").strip().lower()
            if cc_status in {"stunned", "other_cc", "cc_present_type_unknown"}:
                cc_count += 1
            for sk in "QWER":
                human = (row.get(f"{sk}_human_label") or "").strip().lower()
                if human not in LABELS and human != "":
                    raise ValueError(f"Invalid human label for {name} {sk}: {human}")
                if not human:
                    continue
                if human == "unknown":
                    human_unknown += 1
                    continue
                a = old_lookup[name]["abilities"][sk]["visual_state_candidate"]
                b = new_lookup[name]["abilities"][sk]["visual_state_candidate"]
                comparisons.append({
                    "file": name, "timecode": row.get("source_timecode"),
                    "skill": sk, "human": human, "before": PRED.get(a, "unknown"),
                    "after": PRED.get(b, "unknown"),
                    "zero_score": new_lookup[name]["abilities"][sk]["level_zero_score"],
                    "cc_status": cc_status or "unknown",
                })

    def score_field(key, records):
        attempted = [r for r in records if r[key] != "unknown"]
        correct = [r for r in attempted if r[key] == r["human"]]
        return {
            "labeled_pairs": len(records),
            "decided_pairs": len(attempted),
            "correct_when_decided": len(correct),
            "coverage": round(len(attempted) / len(records), 4) if records else None,
            "accuracy_when_decided": round(len(correct) / len(attempted), 4) if attempted else None,
            "correct_over_all_labeled": round(len(correct) / len(records), 4) if records else None,
        }
    before = score_field("before", comparisons)
    after = score_field("after", comparisons)
    breakdown = {
        sk: {"before": score_field("before", [r for r in comparisons if r["skill"] == sk]),
             "after": score_field("after", [r for r in comparisons if r["skill"] == sk])}
        for sk in "QWER"
    }
    return {
        "version": "hud-skill-zero-eval-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene_id": revised.get("scene_id", folder.name),
        "zero_threshold": threshold,
        "manual_frame_count": len(seen),
        "manual_unknown_pairs_excluded": human_unknown,
        "manual_cc_annotated_frames": cc_count,
        "before": before, "after": after, "by_skill": breakdown,
        "new_not_learned_correct": sum(r["before"] != "not_learned" and r["after"] == "not_learned" and r["human"] == "not_learned" for r in comparisons),
        "new_not_learned_wrong": sum(r["before"] != "not_learned" and r["after"] == "not_learned" and r["human"] != "not_learned" for r in comparisons),
        "changes": [r for r in comparisons if r["before"] != r["after"]],
        "caution": (
            "This evaluation uses selected human-labeled HUD images only. "
            "A 0-like glyph is a candidate, not verified ground truth. "
            "No inference about cast timing, precise cooldown, or whether CC is present. "
            "Other replay HUD scales/themes must be calibrated separately."
        ),
    }


def analyze_scene(folder, zero_threshold):
    source = folder / "hud_state_candidates.json"
    if not source.is_file():
        raise FileNotFoundError(f"Original detection JSON missing: {source}")
    orig = json.loads(source.read_text(encoding="utf-8-sig"))
    records = orig.get("frames", [])
    if not records:
        raise ValueError(f"No frames in {source}")
    names = [x["ability_row_image"] for x in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image filenames in input")
    ref = zero_reference()
    revised = copy.deepcopy(orig)
    revised["version"] = "hud-temporal-visual-candidates-with-level-zero-v1"
    revised["source_original_predictions"] = str(source)
    revised["generated_at"] = datetime.now().isoformat(timespec="seconds")
    revised["original_visual_states_preserved_per_skill"] = True
    revised["skill_level_detection_scope"] = "only high-confidence ZERO; nonzero levels not yet OCR-recognized"
    revised["interpretation_policy"] = dict(orig.get("interpretation_policy", {}))
    revised["interpretation_policy"]["not_learned_level_zero_candidate"] = (
        "Bright lower glyph matches 0, and central icon is dark; candidate unlearned. "
        "CC state is still a separate, independent signal."
    )
    states = list(revised.get("allowed_states", []))
    if "not_learned_level_zero_candidate" not in states:
        states.append("not_learned_level_zero_candidate")
    revised["allowed_states"] = states
    counts = Counter()
    reviews, conflicts = [], []
    for fr in revised["frames"]:
        image_file = fr["ability_row_image"]
        if Path(image_file).name != image_file:
            raise ValueError(f"Invalid image filename: {image_file}")
        im = cv2.imread(str(folder / image_file))
        if im is None:
            raise FileNotFoundError(f"HUD PNG missing: {folder / image_file}")
        for sk in "QWER":
            a = fr["abilities"][sk]
            score = matching_score(cut_mask(im, sk), ref)
            old = a["visual_state_candidate"]
            is_zero = bool(score >= zero_threshold)
            new = old
            conflict = False
            if is_zero and old in {"dark_icon_unknown", "unknown"}:
                new = "not_learned_level_zero_candidate"
                counts[f"{sk}_new_not_learned"] += 1
            elif is_zero and old in {"cooldown_overlay_candidate", "colored_icon_candidate", "timer_overlay_unconfirmed"}:
                # A possible layout mismatch / bad template match: don't override.
                conflict = True
                conflicts.append({"file": image_file, "skill": sk, "zero_score": score,
                                  "visual_state": old})
            a["raw_visual_state_candidate"] = old
            a["visual_state_candidate"] = new
            a["level_zero_score"] = score
            a["level_zero_candidate"] = is_zero
            a["skill_level"] = 0 if is_zero and not conflict else None
            a["skill_level_source"] = "opencv_zero_template_candidate" if is_zero else None
            a["skill_level_verified_by_human"] = False
            a["level_zero_conflict"] = conflict
            if new != old:
                a["evidence"] = a.get("evidence", "") + "; lower skill level glyph resembles 0 (NOT a cast)"
            reviews.append({
                "source_timecode": fr.get("source_timecode"),
                "ability_row_image": image_file, "skill": sk,
                "original_visual_candidate": old,
                "zero_match_score": score, "level_zero_candidate": str(is_zero).lower(),
                "new_visual_candidate": new,
                "manual_check_needed": str(conflict or (zero_threshold - 0.08 <= score < zero_threshold + 0.03)).lower(),
            })
    revised["skill_level_summary"] = {
        "frame_count": len(records), "threshold": zero_threshold,
        "new_not_learned_by_skill": {sk: counts.get(f"{sk}_new_not_learned", 0) for sk in "QWER"},
        "zero_conflicts": conflicts,
        "limitations": [
            "Calibrated for 1282x832 replay with 900x132 enlarged ability_row; changed HUD layout may be invalid.",
            "Level-zero template is pixel evidence, not an independent annotation.",
            "Other levels 1-5 are not classified; a failed zero match means unknown, NOT learned.",
            "Stun, silence and other action lockout are unrelated to learned/cooldown state.",
            "Skill casts and exact cooldown seconds are NOT inferred.",
        ],
    }
    out_json = folder / "hud_state_candidates_with_levels.json"
    safe_write_json(out_json, revised)
    review_path = folder / "hud_skill_levels_review.csv"
    with review_path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    evaluation = evaluate_manual(folder, orig, revised, zero_threshold)
    if evaluation is not None:
        safe_write_json(folder / "hud_level_evaluation.json", evaluation)
    return {"scene_id": revised.get("scene_id", folder.name),
            "frames": len(records),
            "new_not_learned": revised["skill_level_summary"]["new_not_learned_by_skill"],
            "conflicts": len(conflicts), "evaluation": (
                {"before": evaluation["before"], "after": evaluation["after"],
                 "new_correct": evaluation["new_not_learned_correct"],
                 "new_wrong": evaluation["new_not_learned_wrong"]}
                if evaluation else "manual labels missing; no evaluation")}


def main():
    p = argparse.ArgumentParser(description="Detect zero level (not-learned candidate) in LoL HUD ability crops; offline")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--only", default="qiyana_candidate_01_scene_01")
    p.add_argument("--zero-threshold", type=float, default=.67)
    args = p.parse_args()
    if not 0.5 <= args.zero_threshold <= 0.95 or not math.isfinite(args.zero_threshold):
        p.error("--zero-threshold must be between 0.5 and 0.95")
    folder = args.base.expanduser().resolve() / args.only
    report = analyze_scene(folder, args.zero_threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", folder / "hud_state_candidates_with_levels.json")
    print("       ", folder / "hud_skill_levels_review.csv")
    print("       ", folder / "hud_level_evaluation.json", "(when manual labels exist)")
    print("Original labels, HUD state candidates, and CC analysis NOT overwritten.")


if __name__ == "__main__":
    main()
