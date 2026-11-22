"""LoL HUD HP recognition-only PaddleOCR benchmark (PP-OCRv5 mobile).

Run from project root:
  python OCR/hp_paddle_benchmark.py --video videos/LOL_001.mp4

This is an experiment, NOT a verified upgrade. All model outputs remain candidates.
Manual labels are ONLY used for scoring, never supplied to model.predict().
Geometry assumes a 1282 x 832 video with the same LoL spectator HUD layout.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF_W, REF_H = 1282, 832
# Original frame coordinate rectangles, validated visually on scene_01/frame_000.
ROIS = {
    "full": (67, 660, 137, 678),
    "current": (72, 660, 94, 678),
    "maximum": (98, 660, 123, 678),
}


def rect_scale(rect: tuple[int, int, int, int], shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = rect
    return (round(x1 * w / REF_W), round(y1 * h / REF_H),
            round(x2 * w / REF_W), round(y2 * h / REF_H))


def crop(frame, region):
    x1, y1, x2, y2 = rect_scale(ROIS[region], frame.shape)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError(f"Empty ROI: {region} {frame.shape}")
    return roi


def enhance(bgr, mode="color"):
    if mode == "clahe":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2)).apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.resize(bgr, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)


def read_result(res):
    """PaddleOCR 3.x TextRecognition Result.json has {'res': {...}}."""
    try:
        j = res.json
        if callable(j):
            j = j()
        if isinstance(j, str):
            j = json.loads(j)
        if not isinstance(j, dict):
            return "", 0.0
        data = j.get("res", j)
        if not isinstance(data, dict):
            return "", 0.0
        return str(data.get("rec_text", "")), float(data.get("rec_score", 0.0))
    except (TypeError, ValueError, AttributeError):
        return "", 0.0


def infer(model, bgr):
    results = model.predict(input=bgr, batch_size=1)
    if not results:
        return "", 0.0
    return read_result(results[0])


def parse_full(s):
    t = re.sub(r"\s+", "", s or "")
    m = re.fullmatch(r"(\d{1,4})/(\d{2,4})", t)
    if not m:
        return None
    c, m = map(int, m.groups())
    return (c, m) if 0 <= c <= m <= 5000 else None


def parse_number(s, part):
    t = re.sub(r"\s+", "", s or "")
    if re.fullmatch(r"\d{1,4}", t) is None:
        return None
    x = int(t)
    if part == "maximum" and not 100 <= x <= 5000:
        return None
    if part == "current" and x > 5000:
        return None
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "videos" / "LOL_001.mp4")
    ap.add_argument("--targets", type=Path, default=ROOT / "OCR" / "references" / "LOL_001_hp_manual_11.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hp_paddle_benchmark" / "LOL_001")
    ap.add_argument("--model", default="PP-OCRv5_mobile_rec")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    if not args.video.is_file():
        ap.error(f"영상 파일 없음: {args.video}")
    if not args.targets.is_file():
        ap.error(f"기존 11프레임 정답 파일 없음: {args.targets}")
    try:
        from paddleocr import TextRecognition
    except ImportError:
        ap.error("PaddleOCR가 없음. 설치: python -m pip install paddlepaddle paddleocr")

    print(f"PaddleOCR text recognizer loading: {args.model}", flush=True)
    model = TextRecognition(model_name=args.model, device="cpu")
    doc = json.loads(args.targets.read_text(encoding="utf-8-sig"))
    samples = doc.get("samples", [])
    if args.only:
        samples = [s for s in samples if s["scene_id"] == args.only]
    if not samples:
        ap.error("대상 프레임 없음")

    out = args.out.resolve()
    crop_dir = out / "inputs"
    crop_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video.resolve()))
    if not cap.isOpened():
        ap.error(f"MP4 열기 실패: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        ap.error("FPS 값이 유효하지 않음")

    rows = []
    try:
        for i, sample in enumerate(samples):
            sec = float(sample["source_timestamp_sec"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(sec * fps))
            ok, frame = cap.read()
            if not ok:
                print(f"[SKIP] frame failed at MP4 {sec}s", flush=True)
                continue
            label = f"{i:02d}_{'scene01' if sample['scene_id'].endswith('scene_01') else 'scene02'}_{sec:.3f}"
            observations = {}
            for region in ("full", "current", "maximum"):
                region_im = crop(frame, region)
                for mode in ("color", "clahe"):
                    key = f"{region}_{mode}"
                    processed = enhance(region_im, mode)
                    cv2.imwrite(str(crop_dir / f"{label}_{key}.png"), processed)
                    text, confidence = infer(model, processed)
                    observations[key] = (text, confidence)

            # Selection is predefined; score means model output confidence,
            # NOT probability the reading is correct. No GT input below.
            full_options = []
            for mode in ("color", "clahe"):
                raw, score = observations[f"full_{mode}"]
                parsed = parse_full(raw)
                if parsed:
                    full_options.append((score, parsed, f"full_{mode}"))
            full_options.sort(reverse=True)
            parts = {}
            for region in ("current", "maximum"):
                opts = []
                for mode in ("color", "clahe"):
                    raw, score = observations[f"{region}_{mode}"]
                    value = parse_number(raw, region)
                    if value is not None:
                        opts.append((score, value, f"{region}_{mode}"))
                parts[region] = max(opts, default=None)

            split_pair = None
            if parts["current"] and parts["maximum"]:
                cur = parts["current"][1]
                max_hp = parts["maximum"][1]
                if cur <= max_hp:
                    split_pair = (cur, max_hp)
            full_pair = full_options[0][1] if full_options else None
            selected = full_pair or split_pair
            if full_pair and split_pair and full_pair != split_pair:
                selected = None  # conflicting methods: abstain
            # Default benchmark output is CANDIDATE only; not for coaching.
            true_pair = (sample.get("manual_hp_current"), sample.get("manual_hp_max"))
            assessed = "REVIEW" if selected is None else ("MATCH_LABEL" if selected == true_pair else "DIFF_FROM_LABEL")
            row = {
                "scene_id": sample["scene_id"], "frame_file": sample["frame_file"],
                "mp4_sec": sec, "model": args.model,
                "full_candidate": "/".join(map(str, full_pair)) if full_pair else "",
                "split_candidate": "/".join(map(str, split_pair)) if split_pair else "",
                "selected_candidate": "/".join(map(str, selected)) if selected else "unknown",
                "label_current": true_pair[0], "label_maximum": true_pair[1],
                "comparison_with_current_labels": assessed,
            }
            for key, (raw, score) in observations.items():
                row[f"{key}_raw"] = raw
                row[f"{key}_score"] = round(score, 4)
            rows.append(row)
            print(f"[{i + 1}/{len(samples)}] {label}: {row['selected_candidate']} ({assessed})", flush=True)
    finally:
        cap.release()

    if not rows:
        ap.error("읽을 수 있는 영상 프레임이 없음")
    with (out / "hp_paddle_benchmark.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (out / "hp_paddle_benchmark.json").write_text(json.dumps({
        "version": "hp-paddle-recognition-only-experiment-v1",
        "created_at": datetime.now().isoformat(),
        "geometry_reference": [1282, 832],
        "roi_coordinates": ROIS,
        "model": args.model,
        "label_policy": "Manual reference used only after model inference, not prompt/model input; 11 frames are development data.",
        "note": "Outputs are candidates; do not automatically pass to Gemini coaching. Verify questionable labels.",
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out / 'hp_paddle_benchmark.csv'}", flush=True)


if __name__ == "__main__":
    main()
