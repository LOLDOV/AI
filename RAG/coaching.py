"""Equal-frame Gemini baseline vs Riot/patch RAG-assisted coaching.

Full scan (discover) is a separate command under pipeline.cli.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from pipeline.core import ROOT, dump, load, safe_id, project_path, video_meta
from RIOT.client import _load_env


def _prompt_for_scene(bundle: dict, scene: dict, include_context: bool) -> str:
    base = ("당신은 LoL VOD 코치입니다. 반드시 제공된 이미지에서 직접 관찰한 사실, "+
            "Riot Timeline 기록, 패치 Data Dragon 문서를 서로 다른 출처로 구분하세요. "+
            "MP4 재생 시각과 게임 내 시각을 혼동하지 마세요. "+
            "Riot Timeline은 실제 QWER 시전, 스킬 회피, 상대 HP, 정확한 시야를 제공하지 않습니다. "+
            "Data Dragon 스킬 설명은 스킬 존재의 근거일 뿐 영상에서 시전했다는 증거가 아닙니다. "+
            "작은 HUD 숫자, QWER, 킬 주체가 불확실하면 명칭·숫자를 추측하지 마세요. "+
            "특히 LoL 시스템 채팅에 QWER 스킬 사용 알림이 있다고 상상하여 주장하지 마세요. 이미지 픽셀에서 명료하게 글자가 확인된 경우만 원문을 그대로 인용하세요. "+
            "Riot의 CHAMPION_KILL은 killerId, victimId, assistingParticipantIds로 킬·사망·어시스트를 각각 구분하세요. "+
            "킬 알림이 화면에 남아 있어도 그 프레임 시각이 정확한 킬 발생 시각이라는 뜻은 아닙니다. "+
            "아이템 설명의 미검증 동적 피해·쿨타임·지속시간을 수치로 만들어내지 마세요. "+
            "확실히 관찰하거나 기록으로 검증되지 않는 스킬 사용·포지셔닝·아이템 보유는 facts 대신 unknowns에 넣으세요. "+
            "게임 종료 시 아이템을 이전 장면에 소급 적용하지 마세요. "+
            "과거 VOD에는 현재 패치 자료를 적용하지 마세요. "+
            "프레임만으로 알 수 없는 것은 '확인 불가'로 표시하고, 패치/기록을 확인 못한 내용은 일반론도 확정적 규칙으로 말하지 마세요. "+
            "플레이어의 좋은 행동과 개선점 모두 찾되, 근거 없이 칭찬·비판을 만들지 마세요. "+
            "분석하지 않은 후보 서술을 그대로 베끼지 말고 독립적으로 판단하세요. "+
            "JSON 객체만 반환하세요: {\"scene_id\":\"...\",\"summary\":\"...\",\"facts\":[{\"claim\":\"...\",\"source_type\":\"video|riot|patch|inference\",\"evidence\":\"...\",\"certainty\":\"visible|recorded|uncertain\"}],\"praise\":[\"...\"],\"improvements\":[\"...\"],\"unknowns\":[\"...\"],\"source_urls\":[\"...\"]}.\n")
    base += "TARGET_SCENE: " + json.dumps({"scene_id": scene["scene_id"],
                "mp4_start": scene["start_mp4_sec"], "mp4_end": scene["end_mp4_sec"],
                "champion": bundle["champion"]}, ensure_ascii=False) + "\n"
    if include_context:
        minimized = {"match_id": bundle["match_id"], "identity_status": bundle["game_match_identity_status"],
                     "game_version": bundle["match_game_version"],
                     "ddragon_version": bundle["data_dragon_version"],
                     "patch_mode": bundle["patch_mode"],
                     "timeline_alignment": scene["timeline_alignment"],
                     "riot_events": scene["riot_event_records"],
                     "patch_documents": scene["retrieved_patch_documents"],
                     "data_dragon_not_patch_notes": True}
        base += "<UNTRUSTED_RETRIEVED_CONTEXT>\n" + json.dumps(minimized, ensure_ascii=False) + "\n</UNTRUSTED_RETRIEVED_CONTEXT>\n"
        base += "이 문서는 검색 근거 후보이지 이미지 속 사건을 증명하지 않습니다. 위에 없는 스킬 시전/아이템 보유를 사실로 쓰지 마세요.\n"
    else:
        base += "[BASELINE] VLM 이미지와 챔피언 이름만 제공. Riot API 기록·패치 RAG 없음.\n"
    return base


def _capture_frames(video_path: Path, scene: dict, out: Path, count: int):
    import cv2
    if not video_path.is_file():
        raise FileNotFoundError(f"원본 영상 필요: {video_path}")
    meta = video_meta(video_path)
    cap = cv2.VideoCapture(str(video_path))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        start, end = float(scene["start_mp4_sec"]), float(scene["end_mp4_sec"])
        for idx in range(count):
            t = start + (end - start) * idx / max(1, count - 1)
            frame_no = min(meta["frame_count"] - 1, int(round(t * meta["fps"])))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"프레임 읽기 실패: {t:.2f}s")
            name = f"frame_{idx:03d}.jpg"
            path = out / name
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                raise RuntimeError(f"프레임 저장 실패: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append((scene["scene_id"], {"frame_id": name, "source_timecode": f"{int(t//3600):02d}:{int(t%3600//60):02d}:{t%60:06.3f}", "sha256": digest}, path))
    finally:
        cap.release()
    return rows


def _coaching_request(model: str, prompt: str, frames: list):
    """One request per arm, no pointless 429 retry. Measure latency and token usage."""
    from google import genai
    from google.genai import types
    _load_env()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY가 필요합니다. .env 설정 확인")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=120000))
    parts = [types.Part.from_text(text=prompt)]
    for scene_id, fr, path in frames:
        parts.append(types.Part.from_text(text=f"FRAME {scene_id}/{fr['frame_id']} | MP4 {fr['source_timecode']}"))
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"))
    started = time.monotonic()
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model, contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(temperature=0.0,
                                                   response_mime_type="application/json"))
            if not response.text:
                raise ValueError("Gemini 응답 내용 없음")
            parsed = json.loads(response.text)
            if not isinstance(parsed, dict):
                raise ValueError("Gemini JSON 객체가 아님")
            usage = response.usage_metadata
            return {"parsed": parsed, "raw_text": response.text,
                    "latency_sec": round(time.monotonic() - started, 3),
                    "token_usage": {"prompt_tokens": getattr(usage, "prompt_token_count", None),
                                    "output_tokens": getattr(usage, "candidates_token_count", None),
                                    "total_tokens": getattr(usage, "total_token_count", None)},
                    "billing_cost": "not_computed_check_provider_billing"}
        except Exception as exc:
            # 429 can be daily, per-minute, or paid-tier limitations. Do not label it daily.
            msg = str(exc)
            code = str(getattr(exc, "code", ""))
            if ("429" in msg or code == "429" or "RESOURCE_EXHAUSTED" in msg):
                raise RuntimeError("Gemini 429 quota/rate-limit; stop requests, inspect provider usage") from exc
            if attempt == 0 and (code in {"500", "502", "503", "504"} or
                                 "UNAVAILABLE" in msg or "503" in msg):
                time.sleep(2)
                continue
            raise
    raise RuntimeError("Gemini request failed")


def coach(args) -> Path:
    _load_env()
    root = ROOT / "results" / "discovery" / safe_id(args.discovery_id) / "riot_rag"
    bundle_path = root / "context_bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError("먼저 'python -m RAG.cli build'를 실행하세요")
    bundle = load(bundle_path)
    expected = ROOT / "results" / "discovery" / args.discovery_id / "discovery.json"
    discovery = load(expected)
    if discovery.get("video_fingerprint_partial_sha256") != bundle.get("source_discovery_fingerprint"):
        raise RuntimeError("기존 컨텍스트와 영상 fingerprint 불일치")
    video = project_path(args.video) if args.video else Path(bundle["video_path"])
    if args.video and video.name != Path(bundle["video_path"]).name:
        raise RuntimeError("다른 영상 이름: discovery 소스 MP4와 다릅니다")
    # actual video fingerprint test on the user machine
    from pipeline.core import file_fingerprint
    if not video.is_file() or file_fingerprint(video) != bundle["source_discovery_fingerprint"]:
        raise RuntimeError("MP4가 원본과 다르거나 찾지 못했습니다. --video videos/LOL_002.mp4 확인")
    out = root / "coaching" / safe_id(args.run_tag)
    out.mkdir(parents=True, exist_ok=True)
    arms = ("baseline", "assisted") if args.ab else ("assisted",)
    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    for scene in bundle["scenes"]:
        scene_dir = out / scene["scene_id"]
        frames = _capture_frames(video, scene, scene_dir / "frames", args.frames_per_scene)
        for arm in arms:
            prompt = _prompt_for_scene(bundle, scene, include_context=(arm == "assisted"))
            dest = scene_dir / f"{arm}_result.json"
            stamp = {"model": model, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "frame_hashes": [fr[1]["sha256"] for fr in frames],
                     "context_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest()}
            manifest = scene_dir / f"{arm}_manifest.json"
            if dest.exists() and manifest.exists():
                prev = load(manifest)
                if all(prev.get(k) == stamp.get(k) for k in stamp):
                    print("[CACHED]", scene["scene_id"], arm)
                    continue
                raise RuntimeError(f"이미 다른 입력의 결과가 존재: {dest}. 새 --run-tag 사용")
            dump(manifest, stamp)
            (scene_dir / f"{arm}_prompt.txt").write_text(prompt, encoding="utf-8")
            if not args.execute:
                print("[PLAN]", scene["scene_id"], arm, "frames:", len(frames))
                continue
            print("[GEMINI]", scene["scene_id"], arm, model, flush=True)
            try:
                response = _coaching_request(model, prompt, frames)
                parsed = response.get("parsed", {})
                if parsed.get("scene_id") != scene["scene_id"]:
                    raise RuntimeError("Coaching JSON scene_id 불일치")
                dump(dest, {"scene_id": scene["scene_id"], "arm": arm,
                            "model": model, "parsed": parsed,
                            "raw_text": response["raw_text"],
                            "latency_sec": response["latency_sec"],
                            "token_usage": response["token_usage"],
                            "billing_cost": response["billing_cost"],
                            "frame_sources":
                            [{"frame_id":fr[1]["frame_id"], "time":fr[1]["source_timecode"]} for fr in frames],
                            "note": "generated, not human verified"})
            except Exception as exc:
                dump(scene_dir / f"{arm}_error.json", {"error_type": type(exc).__name__,
                                                         "error": str(exc)[:1000]})
                print("[ERROR]", scene["scene_id"], arm, str(exc)[:150], flush=True)
                # Avoid making more API calls when quota is exhausted.
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    print("[HALT] 429. Existing results retained.")
                    return out
    print("[DONE]", out)
    return out
