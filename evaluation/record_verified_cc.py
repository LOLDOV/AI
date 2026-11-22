"""Save independently human-verified CC frame annotations without touching Q/W/E/R labels.

Example (the user has directly confirmed CC was present at both timecodes):
    python OpenCV/record_verified_cc.py --only qiyana_candidate_01_scene_02 \
        --times 00:04:24.000 00:04:24.500 --status cc_present_type_unknown \
        --notes "원본 영상 직접 확인: CC 존재, 종류는 미확인"

Supports timestamps NOT preselected into existing hud_manual_validation.csv.
Always backs up an existing CSV with a timestamped .bak filename.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "pipeline" / "LOL_001" / "hud_temporal"
ALLOWED = {"stunned", "other_cc", "cc_present_type_unknown", "none"}


def mark(folder: Path, times: list[str], status: str, notes: str) -> tuple[Path, Path | None]:
    if status not in ALLOWED:
        raise ValueError(f"Bad --status: {status}")
    source = folder / "hud_state_candidates.json"
    records = json.loads(source.read_text(encoding="utf-8-sig"))["frames"]
    lookup = {r["source_timecode"]: r["ability_row_image"] for r in records}
    absent = sorted(set(times) - set(lookup))
    if absent:
        raise ValueError(f"Requested times not found in input HUD frames: {absent}")
    target = folder / "hud_manual_validation.csv"
    original: list[dict] = []
    if target.exists():
        with target.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            original = list(reader)
    else:
        columns = ["source_timecode", "ability_row_image",
                   *[f"{s}_human_label" for s in "QWER"], "notes"]
    for column in ("cc_status", "cc_notes"):
        if column not in columns:
            columns.append(column)
    by_image = {}
    for row in original:
        image = row.get("ability_row_image", "")
        if image in by_image:
            raise ValueError(f"Duplicate CSV entry: {image}")
        by_image[image] = dict(row)
    for stamp in times:
        image = lookup[stamp]
        row = by_image.setdefault(image, {c: "" for c in columns})
        row["source_timecode"] = stamp
        row["ability_row_image"] = image
        row["cc_status"] = status
        row["cc_notes"] = notes
    order = {r["ability_row_image"]: i for i, r in enumerate(records)}
    ordered = sorted(by_image.values(), key=lambda x: order.get(x.get("ability_row_image", ""), 999999))
    backup = None
    if target.exists():
        backup = target.with_name(target.name + "." + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
        shutil.copy2(target, backup)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=folder,
                                     prefix="hud_manual_validation_", suffix=".tmp", delete=False) as out:
        tmp = Path(out.name)
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(tmp, target)
    return target, backup


def main() -> None:
    p = argparse.ArgumentParser(description="사람이 확인한 CC를 CSV에 안전하게 보존")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--only", required=True)
    p.add_argument("--times", nargs="+", required=True)
    p.add_argument("--status", choices=sorted(ALLOWED), default="cc_present_type_unknown")
    p.add_argument("--notes", default="원본 영상 직접 확인: CC 존재, 세부 유형 미확인")
    args = p.parse_args()
    out, bak = mark(args.base.resolve() / args.only, args.times, args.status, args.notes)
    print("CC 수작업 기록 저장:", out)
    print("기존 라벨 백업:", bak or "기존 파일 없음")
    print("추가 프레임:", ", ".join(args.times), "/", args.status)
    print("※ 스킬 정답 Q/W/E/R과 기존 CC 라벨은 덮어쓰지 않았습니다")


if __name__ == "__main__":
    main()
