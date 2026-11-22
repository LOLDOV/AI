"""Evaluate Q/W/E/R HUD predictions with independently annotated crowd-control context.

Usage (vlm-project root):
    python OpenCV/evaluate_hud_predictions.py --only qiyana_candidate_01_scene_01

No API calls. Backwards compatible with CSV files without cc_status.
Important: CC (stun/other_cc) describes character control, NOT a skill cooldown.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "pipeline" / "LOL_001" / "hud_temporal"
PRED = {
    "cooldown_overlay_candidate": "cooldown",
    "colored_icon_candidate": "ready",
    "dark_icon_unknown": "unknown",
    "timer_overlay_unconfirmed": "unknown",
    "unknown": "unknown",
}
LABELS = {"ready", "cooldown", "not_learned", "unknown"}
CC = {"stunned", "other_cc", "cc_present_type_unknown", "none", "unknown", ""}


def pct(num, den):
    return round(num / den, 4) if den else None


def score(records):
    decided = [x for x in records if x["prediction"] != "unknown"]
    correct = [x for x in decided if x["prediction"] == x["ground_truth"]]
    confusion = Counter(f'{x["ground_truth"]} -> {x["prediction"]}' for x in records)
    return {
        "labeled_pairs": len(records),
        "decided_pairs": len(decided),
        "correct_when_decided": len(correct),
        "coverage": pct(len(decided), len(records)),
        "accuracy_when_decided": pct(len(correct), len(decided)),
        "accuracy_including_unknown_as_miss": pct(len(correct), len(records)),
        "confusion_counts": dict(sorted(confusion.items())),
    }


def evaluate(folder: Path, scene_id: str) -> dict:
    pred_file = folder / "hud_state_candidates.json"
    csv_file = folder / "hud_manual_validation.csv"
    if not pred_file.is_file() or not csv_file.is_file():
        raise FileNotFoundError(f"필요한 두 파일이 없음: {pred_file}, {csv_file}")

    preds = json.loads(pred_file.read_text(encoding="utf-8-sig"))
    lookup = {x["ability_row_image"]: x for x in preds.get("frames", [])}
    comparisons = []
    abstained_human = []
    missing = []
    cc_rows = []
    seen_rows = set()
    with csv_file.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            filename = (row.get("ability_row_image") or "").strip()
            if not filename:
                continue
            if filename not in lookup:
                missing.append(filename)
                continue
            if filename in seen_rows:
                raise ValueError(f"CSV에 중복 프레임: {filename}")
            seen_rows.add(filename)
            cc = (row.get("cc_status") or "").strip().lower()
            if cc not in CC:
                raise ValueError(f"{filename}: 잘못된 cc_status '{cc}' (stunned/other_cc/cc_present_type_unknown/none/unknown/빈칸)")
            cc = cc if cc in {"stunned", "other_cc", "cc_present_type_unknown", "none"} else "unknown"
            data = lookup[filename]
            if cc in {"stunned", "other_cc", "cc_present_type_unknown"}:
                cc_rows.append({
                    "frame_file": filename,
                    "timecode": row.get("source_timecode", ""),
                    "cc_status": cc,
                    "cc_notes": row.get("cc_notes", ""),
                    "warning": "CC 중 스킬 사용 불가와 스킬 쿨타임을 구분해야 함",
                })
            for sk in "QWER":
                human = (row.get(f"{sk}_human_label") or "").strip().lower()
                if human not in LABELS and human != "":
                    raise ValueError(f"{filename} {sk}: 잘못된 정답 '{human}'")
                if not human:
                    continue
                raw = data["abilities"][sk]["visual_state_candidate"]
                comparison = {
                    "frame_file": filename,
                    "timecode": row.get("source_timecode", ""),
                    "skill": sk,
                    "ground_truth": human,
                    "visual_candidate": raw,
                    "prediction": PRED.get(raw, "unknown"),
                    "cc_status": cc,
                    "matches": None if human == "unknown" else PRED.get(raw, "unknown") == human,
                }
                if human == "unknown":
                    abstained_human.append(comparison)
                else:
                    comparisons.append(comparison)
    all_counts = score(comparisons)
    report = {
        "version": "hud-eval-with-cc-v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scene_id": scene_id,
        **all_counts,
        "labeled_skill_frame_pairs": all_counts["labeled_pairs"],  # Backward compatible keys
        "human_unknown_pairs_excluded": len(abstained_human),
        "annotated_cc_frames": len(cc_rows),
        "missing_prediction_frames": missing,
        "by_skill": {sk: score([x for x in comparisons if x["skill"] == sk]) for sk in "QWER"},
        "by_cc_context": {
            "cc_confirmed": score([x for x in comparisons if x["cc_status"] in {"stunned", "other_cc", "cc_present_type_unknown"}]),
            "no_cc_confirmed": score([x for x in comparisons if x["cc_status"] == "none"]),
            "cc_not_checked": score([x for x in comparisons if x["cc_status"] == "unknown"]),
        },
        "cc_review": cc_rows,
        "human_unknown_records": abstained_human,
        "comparisons": comparisons,
        "caution": (
            "사람이 선택한 프레임에 한정된 검증; 전체 영상 정확도와 다름. "
            "스킬 쿨타임과 스턴 상태는 독립 변수이며, 라벨 unknown은 정확도 모수에서 제외. "
            "CC는 사람의 관찰을 기록한 것이지 자동 검출 결과가 아님."
        ),
    }
    output = folder / "hud_manual_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description="HUD 예측값과 사람 라벨+CC 상황을 평가 (API 없음)")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--only", default="qiyana_candidate_01_scene_01")
    args = p.parse_args()
    folder = args.base.resolve() / args.only
    report = evaluate(folder, args.only)
    summary_keys = ["scene_id", "labeled_skill_frame_pairs", "decided_pairs",
                    "correct_when_decided", "coverage", "accuracy_when_decided",
                    "accuracy_including_unknown_as_miss", "human_unknown_pairs_excluded",
                    "annotated_cc_frames"]
    print(json.dumps({k:report[k] for k in summary_keys}, ensure_ascii=False, indent=2))
    print("스킬별:", json.dumps({k:{"coverage":v["coverage"],"accuracy_when_decided":v["accuracy_when_decided"]}
                             for k,v in report["by_skill"].items()}, ensure_ascii=False))
    print("저장:", folder / "hud_manual_evaluation.json")


if __name__ == "__main__":
    main()
