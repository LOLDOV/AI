"""Candidate-driven full-frame CC review for LoL VOD.

Designed to follow OpenCV/hud_cc_context.py without changing its predictions.
Default: only extract full-frame JPEG evidence (local / no Gemini API).
Optional --run: ask Gemini to assess *visible CC cues* for the selected champion.

From vlm-project root:
    python gemini/verify_cc_fullscreen.py --only qiyana_candidate_01_scene_01
    python gemini/verify_cc_fullscreen.py --only qiyana_candidate_01_scene_01 --run

Requires cv2 (opencv-python) and for --run google-genai + python-dotenv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "hud_temporal"
DEFAULT_MODEL = "gemini-3.6-flash"

PROMPT = """너는 리그 오브 레전드 리플레이의 CC(군중 제어) **시각 증거 판독기**다.
관심 대상은 좌하단 챔피언 HUD가 가리키는 지정된 분석 챔피언이다.
입력 프레임은 시간순 원본 전체 화면이며, 마지막 프레임은 중앙 확대 보조 이미지이다.
이미지에 등장하는 다른 챔피언의 CC를 **분석 대상 챔피언의 CC로 잘못 귀속하지 말 것**.
입력에 제시된 'HUD 아이콘 변화 후보'는 사실로 확정되지 않았으므로 모델 판단을 유도하지 않는다.

관찰 규칙:
1) 분석 대상 챔피언의 머리 위/체력바 근처에 '기절', '속박', '침묵', '공중에 뜸', '제압', '공포' 등의 상태 문구나 식별 가능한 CC 표시가 있는지 확인.
2) 텍스트가 있다면 정확히 화면에 보이는 글자만 evidence_text에 기재. 없으면 null.
3) 플레이어와 CC 표시의 연결이 불분명하면 uncertain. HUD 아이콘 암전만으로 CC 확정 금지.
4) 프레임 간격 사이의 CC 시작/종료 시각, 상대의 스킬 시전, 피해 원인을 만들어내지 말 것.
5) 눈에 확실히 보이는 증거가 있어도 'VLM 관찰'일 뿐 사람 검증은 아님.

오직 다음 구조의 JSON만 출력:
{
  "target_champion": "지정된 분석 챔피언",
  "visual_status": "visible_cc_on_target | possible_cc_on_target | no_visible_cc_evidence | uncertain",
  "cc_type": "stunned | rooted | silenced | airborne | suppressed | feared | other | unknown",
  "evidence_text": null,
  "evidence_timecodes": ["00:03:15.500"],
  "reason": "관찰 근거 1~2문장",
  "limitations": "한계 1문장"
}
"""


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def timecode(t: float) -> str:
    t = max(0.0, float(t))
    minutes = int(t // 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:{t % 60:06.3f}"


def capture_frame(cap: cv2.VideoCapture, sec_in_clip: float, fps: float, frame_count: int):
    if sec_in_clip < -0.001:
        return None
    index = int(round(sec_in_clip * fps))
    if index >= frame_count or index < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    if not ok:
        return None
    return index, img


def center_zoom(im):
    """A secondary contextual crop, never a substitute for the full screen."""
    height, width = im.shape[:2]
    x1, x2 = int(0.19 * width), int(0.81 * width)
    y1, y2 = int(0.16 * height), int(0.79 * height)
    crop = im[y1:y2, x1:x2]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_CUBIC)


def contact_sheet(photos: list[tuple[str, Any]], destination: Path):
    if not photos:
        return
    import numpy as np
    width, height, cols = 512, 350, 2
    rows = (len(photos) + cols - 1) // cols
    canvas = np.full((rows * height, cols * width, 3), 245, dtype=np.uint8)
    for i, (label, photo) in enumerate(photos):
        h, w = photo.shape[:2]
        neww = width - 16
        newh = int(neww * h / w)
        if newh > height - 40:
            newh = height - 40
            neww = int(newh * w / h)
        small = cv2.resize(photo, (neww, newh), interpolation=cv2.INTER_AREA)
        left = (i % cols) * width + (width - neww) // 2
        top = (i // cols) * height + 25
        canvas[top:top + newh, left:left + neww] = small
        cv2.putText(canvas, label, ((i % cols) * width + 12, (i // cols) * height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (32, 32, 32), 1, cv2.LINE_AA)
    cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def extract_event(cap, fps: float, frame_count: int, event: dict,
                  out_dir: Path, offset: float, half_window: float, step: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = float(event["timestamp_sec"])
    count = int(round(2 * half_window / step))
    seconds = [round(mid - half_window + j * step, 3) for j in range(count + 1)]
    seconds = sorted(set(seconds + [mid]))
    frames: list[dict] = []
    preview: list[tuple[str, Any]] = []
    best_mid = None
    for i, source_s in enumerate(seconds):
        got = capture_frame(cap, source_s - offset, fps, frame_count)
        if got is None:
            continue
        index, img = got
        filename = f"frame_{i:02d}_full.jpg"
        cv2.imwrite(str(out_dir / filename), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append({
            "source_timestamp_sec": source_s,
            "source_timecode": timecode(source_s),
            "video_local_timestamp_sec": round(index / fps, 4),
            "video_frame_index": index,
            "full_frame_file": filename,
        })
        preview.append((timecode(source_s) + "  FULL", img))
        if best_mid is None or abs(source_s - mid) < abs(best_mid[0] - mid):
            best_mid = (source_s, img)
    zoom_filename = None
    if best_mid is not None:
        zoom_filename = "candidate_center_zoom.jpg"
        zoom = center_zoom(best_mid[1])
        cv2.imwrite(str(out_dir / zoom_filename), zoom, [cv2.IMWRITE_JPEG_QUALITY, 92])
        preview.append((timecode(best_mid[0]) + "  CENTER ZOOM", zoom))
    contact_sheet(preview, out_dir / "contact_sheet.jpg")
    return {"event_timecode": event.get("timecode"),
            "event_timestamp_sec": mid,
            "source": "OpenCV HUD visual transition candidate; NOT an established CC",
            "hud_visual_evidence": {"darkened_slots": event.get("darkened_slots", []),
                                    "recovered_slots_within_window": event.get("recovered_slots_within_window", [])},
            "screenshots": frames,
            "center_zoom_file": zoom_filename,
            "snapshot_count": len(frames),
            "source_video_offset_seconds": offset,
            "model_cc_observation_file": "cc_vlm_observation.json" if frames else None}


def parse_json_response(s: str) -> dict:
    raw = s.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("JSON 응답이 object가 아닙니다")
    allowed = {"visible_cc_on_target", "possible_cc_on_target", "no_visible_cc_evidence", "uncertain"}
    status = obj.get("visual_status")
    if status not in allowed:
        raise ValueError(f"알 수 없는 visual_status: {status!r}")
    obj["verified_by_human"] = False
    obj["observation_source"] = "gemini_full_frame_unverified"
    obj["cc_confirmed"] = False  # A model result is not ground truth
    return obj


def gemini_review(event_dir: Path, manifest: dict, model: str, champion: str) -> dict:
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(".env에 GEMINI_API_KEY가 없습니다. --run 없이 실행하면 이미지 추출만 합니다")
    parts = [types.Part.from_text(text=PROMPT + "\n\n대상 영상 시각: " + str(manifest["event_timecode"]) + "\n분석 대상 챔피언: " + champion)]
    for frame in manifest["screenshots"]:
        parts.append(types.Part.from_text(text="원본 전체 화면 " + frame["source_timecode"]))
        parts.append(types.Part.from_bytes(data=(event_dir / frame["full_frame_file"]).read_bytes(), mime_type="image/jpeg"))
    if manifest.get("center_zoom_file"):
        parts.append(types.Part.from_text(text="위 원본 영상에서 이벤트 중심 시점의 중앙 영역을 확대한 보조 화면. 대상이 중앙에 없을 수도 있습니다"))
        parts.append(types.Part.from_bytes(data=(event_dir / manifest["center_zoom_file"]).read_bytes(), mime_type="image/jpeg"))
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=120000))
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    output = getattr(response, "text", None)
    if not output:
        raise ValueError("Gemini 응답이 비었습니다")
    result = parse_json_response(output)
    result["model"] = model
    result["target_champion"] = champion
    result["event_timecode"] = manifest["event_timecode"]
    return result


def main():
    ap = argparse.ArgumentParser(description="Full-screen CC evidence extraction + optional Gemini review")
    ap.add_argument("--base", type=Path, default=None, help="Results root containing video ID folders; defaults to results/pipeline")
    ap.add_argument("--only", default="qiyana_candidate_01_scene_01", help="scene_id")
    ap.add_argument("--video-id", default="LOL_001", help="video id used under results/hud_temporal/ and videos/")
    ap.add_argument("--champion", default="Qiyana", help="champion being reviewed, default Qiyana")
    ap.add_argument("--video", type=Path, default=None, help="source recording MP4; default videos/<video-id>.mp4")
    ap.add_argument("--video-offset-sec", type=float, default=0.0,
                    help="MP4 local 0 sec corresponds to this time in source timeline; use 0 for full LOL_001.mp4")
    ap.add_argument("--half-window-sec", type=float, default=1.0)
    ap.add_argument("--step-sec", type=float, default=0.5)
    ap.add_argument("--run", action="store_true", help="Call Gemini API; may incur charges")
    ap.add_argument("--overwrite", action="store_true", help="Redo an existing VLM result")
    ap.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()
    if not 0.1 <= args.step_sec <= 3 or not 0 <= args.half_window_sec <= 10:
        ap.error("--step-sec must be 0.1..3, --half-window-sec must be 0..10")
    scene_dir = (args.base or (ROOT / 'results' / 'pipeline')).resolve() / args.video_id / 'hud_temporal' / args.only
    context_file = scene_dir / "hud_cc_context.json"
    if not context_file.is_file():
        raise FileNotFoundError(f"CC 데이터 없음: {context_file}. 먼저 python OpenCV/hud_cc_context.py --only {args.only}")
    cc_data = read_json(context_file)
    candidates = cc_data.get("candidate_events", [])
    if not candidates:
        print("분석할 CC 시각 후보가 없습니다. 참고: 후보 0건은 CC가 없었다는 증거가 아닙니다")
        return
    video_path = (args.video or (ROOT / "videos" / f"{args.video_id}.mp4")).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"MP4 없음: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or fps <= 0 or frame_count <= 0:
            raise RuntimeError("영상을 열 수 없거나 fps/frame_count를 읽지 못했습니다")
        review_dir = scene_dir / "cc_fullscreen_review"
        review_dir.mkdir(parents=True, exist_ok=True)
        result = {"version": "cc-fullscreen-review-v1",
                  "created_at": datetime.now().isoformat(timespec="seconds"),
                  "scene_id": args.only,
                  "video_id": args.video_id,
                  "target_champion": args.champion,
                  "source_cc_context": str(context_file),
                  "video": str(video_path),
                  "fps": fps,
                  "video_offset_sec": args.video_offset_sec,
                  "mode": "gemini_api" if args.run else "extract_only_no_api",
                  "note": "Model suggestions do not convert hud_cc_context.json cc_confirmed into true.",
                  "events": []}
        for i, event in enumerate(candidates, start=1):
            event_dir = review_dir / f"cc_event_{i:03d}"
            data = extract_event(cap, fps, frame_count, event, event_dir, args.video_offset_sec,
                                 args.half_window_sec, args.step_sec)
            write_json(event_dir / "evidence_manifest.json", data)
            if not data["screenshots"]:
                data["error"] = "No frames extracted: video/local time offset is wrong or event is outside MP4"
                print("[WARN]", data["error"])
            elif args.run:
                output = event_dir / "cc_vlm_observation.json"
                if output.exists() and not args.overwrite:
                    data["model_review_status"] = "skipped_existing_result"
                else:
                    try:
                        obs = gemini_review(event_dir, data, args.model, args.champion)
                        write_json(output, obs)
                        data["model_review_status"] = "saved_unverified_vlm_observation"
                    except Exception as exc:
                        data["model_review_status"] = "api_or_parsing_failed"
                        data["model_error"] = str(exc)[:350]
                        print("[ERROR] Gemini:", data["model_error"])
            result["events"].append(data)
            print(f"[{i}/{len(candidates)}] {event.get('timecode')} - {data['snapshot_count']} full frames -> {event_dir}")
        write_json(review_dir / "cc_fullscreen_review_summary.json", result)
        print("완료:", review_dir / "cc_fullscreen_review_summary.json")
        if not args.run:
            print("※ 이미지 추출만 완료. Gemini API는 호출하지 않았습니다. --run을 붙이면 VLM 관찰 결과가 추가됩니다")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
