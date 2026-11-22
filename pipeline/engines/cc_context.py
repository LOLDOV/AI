"""LoL HUD-based CC *review candidate* detector, v2.

It never classifies a CC type or determines CC is absent.
Fixes false negatives when two colorful icons go dark together, but only one
returns to color while the other displays a countdown (scene 02 case).
The previous v1 required BOTH icons to return to color.

Run from the project root:
    python OpenCV/hud_cc_context.py --only qiyana_candidate_01_scene_01
    python OpenCV/hud_cc_context.py --only qiyana_candidate_01_scene_02

Inputs: hud_state_candidates.json, optional hud_manual_validation.csv
Outputs: hud_cc_context.json, hud_cc_review.csv
No API calls. Does not overwrite human labels, level predictions or video data.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "hud_temporal" / "LOL_001"
COLOR = "colored_icon_candidate"
DARK = "dark_icon_unknown"
COOLDOWN_LIKE = {"cooldown_overlay_candidate", "timer_overlay_unconfirmed"}
HUMAN_POSITIVE = {"stunned", "other_cc", "cc_present_type_unknown"}
HUMAN_ALLOWED = HUMAN_POSITIVE | {"none", "unknown", ""}
SKILLS = "QWE"  # R unlearned/dark is not evidence of a new action lockout


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def _state(frame: dict, skill: str) -> str:
    # Raw optical states only: level-zero patch is deliberately kept independent.
    return frame["abilities"][skill]["visual_state_candidate"]


def _human_labels(folder: Path) -> dict[str, dict]:
    csv_path = folder / "hud_manual_validation.csv"
    found: dict[str, dict] = {}
    if not csv_path.exists():
        return found
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("ability_row_image") or "").strip()
            status = (row.get("cc_status") or "").strip().lower()
            if not name:
                continue
            if status not in HUMAN_ALLOWED:
                raise ValueError(f"{csv_path}: invalid cc_status {status!r} for {name}")
            if name in found:
                raise ValueError(f"Duplicate manual entry: {name}")
            if status in HUMAN_POSITIVE | {"none"}:
                found[name] = {
                    "cc_status": status, "cc_notes": row.get("cc_notes") or "",
                    "source": "human_annotation",
                }
    return found


def build(folder: Path, recover_window_s: float = 2.0,
          transition_window_s: float = 0.5) -> dict:
    if not 0 < recover_window_s <= 10 or not 0 <= transition_window_s <= 2:
        raise ValueError("recover_window_s must be in (0,10], transition_window_s in [0,2]")
    pred_path = folder / "hud_state_candidates.json"
    data = _read_json(pred_path)
    frames = sorted(data["frames"], key=lambda v: float(v["source_timestamp_sec"]))
    if not frames:
        raise ValueError(f"No frames in {pred_path}")
    if len({fr["ability_row_image"] for fr in frames}) != len(frames):
        raise ValueError("Duplicate frame image names")
    labels = _human_labels(folder)
    tvals = [float(fr["source_timestamp_sec"]) for fr in frames]
    events: list[dict] = []
    flagged: set[str] = set()
    recent_drops: dict[str, tuple[int, float]] = {}
    last_event_time = float("-inf")

    for i in range(1, len(frames)):
        prev, cur = frames[i - 1], frames[i]
        now_t = tvals[i]
        for sk in SKILLS:
            if _state(prev, sk) == COLOR and _state(cur, sk) == DARK:
                recent_drops[sk] = (i, now_t)
        # Exclude stale independent transitions. A short window tolerates
        # darkening in adjacent 0.5s samples, rather than demanding same frame.
        recent = {sk: rec for sk, rec in recent_drops.items()
                  if 0 <= now_t - rec[1] <= transition_window_s + 1e-6}
        if len(recent) < 2:
            continue
        if now_t - last_event_time < max(transition_window_s, 0.5) - 1e-6:
            continue
        # Avoid reusing old drops for a second event.
        skills = sorted(recent.keys())
        for sk in skills:
            recent_drops.pop(sk, None)
        first_t = min(rec[1] for rec in recent.values())
        starts = {sk: rec[1] for sk, rec in recent.items()}
        last_event_time = now_t

        colored_recovered: list[str] = []
        cooldown_transition: list[str] = []
        other_outcomes: dict[str, str] = {}
        latest_dark = now_t
        for sk in skills:
            start_i = recent[sk][0]
            final_state = "still_dark_or_not_observed"
            for k in range(start_i + 1, len(frames)):
                if tvals[k] - now_t > recover_window_s + 1e-6:
                    break
                nxt = _state(frames[k], sk)
                if nxt == DARK:
                    latest_dark = max(latest_dark, tvals[k])
                    continue
                if nxt == COLOR:
                    colored_recovered.append(sk)
                    final_state = "returned_to_colored"
                elif nxt in COOLDOWN_LIKE:
                    cooldown_transition.append(sk)
                    final_state = "changed_to_timer_or_cooldown_visual"
                else:
                    final_state = "changed_to_other_visual"
                break
            other_outcomes[sk] = final_state

        if len(colored_recovered) >= 2:
            priority, pattern = "high", "multiple_colored_recoveries"
        elif colored_recovered and cooldown_transition:
            priority, pattern = "medium", "partial_colored_recovery_plus_timer_visual"
        elif colored_recovered:
            priority, pattern = "medium", "partial_colored_recovery"
        else:
            priority, pattern = "low", "simultaneous_darkening_only"

        # The visual span is NOT a CC duration. Flag successive dark frames
        # simply as a convenient window for human inspection.
        last_possible_dark = now_t
        for j in range(i, len(frames)):
            if tvals[j] - now_t > recover_window_s + 1e-6:
                break
            if any(_state(frames[j], sk) == DARK for sk in skills):
                flagged.add(frames[j]["ability_row_image"])
                last_possible_dark = tvals[j]
            else:
                break
        flagged.add(cur["ability_row_image"])
        # Include the earlier adjacent frame in case two slots darkened 0.5s apart.
        for sk, rec in recent.items():
            flagged.add(frames[rec[0]]["ability_row_image"])

        human_hits = []
        for j, frame in enumerate(frames):
            if first_t - 1e-6 <= tvals[j] <= last_possible_dark + 1e-6:
                human = labels.get(frame["ability_row_image"])
                if human and human["cc_status"] in HUMAN_POSITIVE:
                    human_hits.append(frame["source_timecode"])
        event = {
            "timecode": cur["source_timecode"],
            "timestamp_sec": now_t,
            "ability_row_image": cur["ability_row_image"],
            "darkened_slots": skills,
            "darkening_times": starts,
            "recovered_slots_within_window": colored_recovered,
            "changed_to_cooldown_visual_within_window": cooldown_transition,
            "post_darkening_visual_outcomes": other_outcomes,
            "candidate": "possible_temporary_action_lockout",
            "candidate_pattern": pattern,
            "review_priority": priority,  # a ranking, NOT a probability
            "candidate_visual_window": {
                "start_sec": first_t, "last_dark_frame_sec": last_possible_dark,
                "is_confirmed_cc_duration": False,
            },
            "cc_confirmed": False,  # never auto-confirm CC from HUD alone
            "human_verified_cc_frame_timecodes": human_hits,
            "human_verified_cc_in_visual_window": bool(human_hits),
            "description": (
                "여러 Q/W/E 아이콘이 가까운 시점에 컬러→암전으로 변화. "
                "전부 컬러로 복귀하지 않아도 후보에 포함. "
                "스턴·침묵 등 CC 또는 다른 HUD 변화 가능; 전체 화면 검증 필요"
            ),
        }
        events.append(event)

    entries = []
    for fr in frames:
        name = fr["ability_row_image"]
        human = labels.get(name)
        status = human["cc_status"] if human else "unknown"
        entries.append({
            "source_timecode": fr["source_timecode"],
            "ability_row_image": name,
            "human_cc_status": status,
            "cc_source": "human_annotation" if human else "visual_candidate_only_or_none",
            "possible_cc_visual_candidate": name in flagged,
            "cooldown_interpretation_policy": "CC visual evidence never proves skill cooldown or cast time",
            "usable_for_cast_now": False if status in HUMAN_POSITIVE else None,
            "notes": human["cc_notes"] if human else "",
        })

    positive = [f for f in entries if f["human_cc_status"] in HUMAN_POSITIVE]
    negative = [f for f in entries if f["human_cc_status"] == "none"]
    evaluation = {
        "human_verified_cc_frames": len(positive),
        "detected_human_cc_frames": sum(f["possible_cc_visual_candidate"] for f in positive),
        "candidate_hit_rate_on_annotated_positive_frames": (
            round(sum(f["possible_cc_visual_candidate"] for f in positive) / len(positive), 4)
            if positive else None
        ),
        "human_verified_no_cc_frames": len(negative),
        "candidate_flags_on_verified_no_cc_frames": sum(f["possible_cc_visual_candidate"] for f in negative),
        "false_positive_rate_on_annotated_negative_frames": (
            round(sum(f["possible_cc_visual_candidate"] for f in negative) / len(negative), 4)
            if negative else None
        ),
        "warning": "Only sparse human-labeled frames were scored; not video-wide CC recall or precision",
    }
    output = {
        "version": "hud-cc-candidate-v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene_id": data.get("scene_id", folder.name),
        "human_verified_cc_frames": len(positive),
        "candidate_events": events,
        "frames": entries,
        "candidate_validation": evaluation,
        "limitations": [
            "검출된 것은 CC 의심 구간이지 CC 자동 확정이 아닙니다.",
            "HUD 암전은 CC 이외의 원인이 있을 수 있습니다. CC가 검출되지 않았다고 CC가 없다는 뜻은 아닙니다.",
            "컬러 복귀와 숫자 오버레이 전환은 각각 별개의 시각 상태이며 스킬 시전 확정이 아닙니다.",
            "숫자 오버레이가 CC 때문에 생겼다고 추론하지 않습니다. CC와 스킬 쿨다운은 별개입니다.",
            "프레임 간격 0.5초보다 짧은 CC는 놓칠 수 있습니다.",
            "CC 유형은 사람 또는 전체 화면 VLM 관찰로 별도 검증해야 합니다.",
        ],
    }
    out = folder / "hud_cc_context.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    review = folder / "hud_cc_review.csv"
    with review.open("w", newline="", encoding="utf-8-sig") as f:
        cols = ["source_timecode", "ability_row_image", "darkened_slots",
                "recovered_slots_within_window", "changed_to_cooldown_visual_within_window",
                "candidate_pattern", "review_priority", "candidate", "cc_confirmed",
                "human_verified_cc_in_visual_window", "review_note"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ev in events:
            w.writerow({
                "source_timecode": ev["timecode"],
                "ability_row_image": ev["ability_row_image"],
                "darkened_slots": "/".join(ev["darkened_slots"]),
                "recovered_slots_within_window": "/".join(ev["recovered_slots_within_window"]),
                "changed_to_cooldown_visual_within_window": "/".join(ev["changed_to_cooldown_visual_within_window"]),
                "candidate_pattern": ev["candidate_pattern"],
                "review_priority": ev["review_priority"],
                "candidate": ev["candidate"],
                "cc_confirmed": False,
                "human_verified_cc_in_visual_window": ev["human_verified_cc_in_visual_window"],
                "review_note": ev["description"],
            })
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="LoL HUD CC 후보 탐지 v2 — 복귀가 일부만 발생해도 후보 기록 (오프라인)")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--only", default="qiyana_candidate_01_scene_01")
    p.add_argument("--recover-window", type=float, default=2.0)
    p.add_argument("--transition-window", type=float, default=0.5)
    args = p.parse_args()
    folder = args.base.resolve() / args.only
    out = build(folder, args.recover_window, args.transition_window)
    print(f"장면 {out['scene_id']}, {len(out['frames'])}장 / CC 의심 후보 {len(out['candidate_events'])}건")
    for ev in out["candidate_events"]:
        print(f"  {ev['timecode']} {','.join(ev['darkened_slots'])} "
              f"pattern={ev['candidate_pattern']} priority={ev['review_priority']} "
              f"human_verified={ev['human_verified_cc_in_visual_window']}")
    print("수작업 CC 확인 프레임:", out["human_verified_cc_frames"])
    print("저장:", folder / "hud_cc_context.json")
    print("저장:", folder / "hud_cc_review.csv")


if __name__ == "__main__":
    main()
