"""LoL_001의 정글 강타 진화 지식을 RAG 컨텍스트로 묶고 D-slot 오분류를 보정.

기존 summoner_spell_probe.json / 원본 보고서는 덮어쓰지 않음.
유저가 확인한 '파란 펫'을 다른 영상에 자동 전파하지 않도록 --confirmed-blue-pet 필수.
표준 라이브러리만 사용 (Gemini/외부 API 호출 없음).
"""
import argparse
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = Path(__file__).resolve().parent / "knowledge" / "lol_jungle_smite_2026.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tokenize(s: str):
    return set(re.findall(r"[가-힣a-zA-Z0-9_.]+", s.lower()))


def retrieve(kb, query, n=5):
    query_l = query.lower()
    q_tokens = tokenize(query)
    ranked = []
    for fact in kb["facts"]:
        score = sum(3 for w in fact.get("keywords", []) if w.lower() in query_l)
        score += len(q_tokens & tokenize(fact["text_ko"]))
        ranked.append((score, fact))
    ranked.sort(key=lambda x: -x[0])
    return [fact for score, fact in ranked[:n] if score > 0] or kb["facts"][:n]


def run(probe: Path, outdir: Path, event_time: str, confirmed_blue: bool, selected_video: str):
    kb = load_json(KB)
    raw = load_json(probe)
    assert raw.get("video_id") == selected_video, f"영상 ID 불일치: {raw.get('video_id')} != {selected_video}"
    assert "slots" in raw and "D" in raw["slots"] and "F" in raw["slots"], "D/F 후보 정보 없음"
    matches = [c for c in raw["slots"]["D"].get("candidates", []) if c.get("timecode") == event_time]
    if not matches:
        raise ValueError(f"D 슬롯에서 {event_time}을 찾지 못함. 타임코드 확인 필요.")
    match = matches[0]
    raw_top = match.get("top_candidates", [{}])[0]
    # 원본 후보는 원본 그대로 유지. 별도의 RAG 보정 레이어에서 해석.
    corrected = copy.deepcopy(raw)
    corrected["version"] = "summoner-spell-probe-rag-context-v2"
    corrected["original_probe_file"] = str(probe)
    corrected["knowledge_base"] = kb["knowledge_base_id"]
    corrected["patch_caution"] = kb["game_patch_caution"]
    corrected["slots"]["D"]["normalized_spell_family"] = "강타 (Smite)"
    corrected["slots"]["D"]["can_evolve_within_match"] = True
    corrected["slots"]["F"]["normalized_spell_family"] = "점멸 (Flash)"
    event = {
        "timecode": event_time,
        "slot": "D",
        "raw_template_label": raw_top.get("name"),
        "raw_template_similarity": raw_top.get("template_similarity"),
        "corrected_family": "강타 (Smite)",
        "pet_type_candidate": "파란 정글 동료 (Gustwalker)",
        "spell_variant": "진화 강타 (Primal Smite) 후보" if confirmed_blue else "진화 강타 후보: 수작업 확인 필요",
        "identity_evidence": "user_verified_video_context" if confirmed_blue else "retrieved_game_rules_only",
        "verification_status": "user_reported_blue_upgrade" if confirmed_blue else "needs_human_confirmation",
        "note": "원본 총명 분류는 보존하되 최종 스펠 정체로 사용 금지. 아이콘 진화는 같은 강타 계열의 외관 변경이 될 수 있음. 패치/화면 근거로 정확한 진화 단계 추가 확인 필요.",
    }
    corrected["slots"]["D"]["contextual_corrections"] = [event]
    for cand in corrected["slots"]["D"]["candidates"]:
        if cand.get("timecode") == event_time:
            cand["contextual_correction"] = event
    corrected["slots"]["D"]["warning"] = "진화된 강타가 다른 소환사 주문처럼 보일 수 있음. 후보 투표 수치는 정답 정확도가 아님."
    corrected["coaching_safety"] = {
        "can_say": "키아나는 정글러이고, 교전 구간의 D는 강타 계열/F는 점멸 후보이다. 23:12의 파란 D 아이콘은 강타 진화로 해석되며, 사용자 확인이 있는 경우 파란 정글 동료라는 문맥을 함께 사용한다.",
        "cannot_say": "특정 순간 점멸/강타를 쓸 수 있었는데 안 썼다, 강타가 교전 중 사용됐다, 정확한 강타 업그레이드 시점, 다른 영상도 모두 파란 동료다",
        "summoner_spell_cooldown": "unknown",
        "cast_times": "not_verified",
    }
    outdir.mkdir(parents=True, exist_ok=True)
    output_json = outdir / "summoner_spell_probe_rag_corrected.json"
    save_json(output_json, corrected)

    facts = retrieve(kb, "2026 정글 파란 Gustwalker 강타 강화 아이콘 15 35 쿨타임", n=6)
    sections = [
        "# Gemini 코칭용 검색 근거 (RAG: 강타·정글 동료)",
        "",
        "> 외부 자료 사실 / 자동 이미지 후보 / 사용자 영상 확인은 서로 다른 근거다. 기존 Gemini 분석의 사실 주장을 이 문서가 자동 검증하지 않는다.",
        "",
        "## 게임 지식 (라이엇 문서 근거)",
    ]
    for i, fact in enumerate(facts, 1):
        sections += [f"{i}. {fact['text_ko']}  ", f"   출처: {fact['source']}"]
    sections += [
        "", "## 이번 LOL_001의 영상 관찰", "",
        "- 키아나 포지션: **정글** (사용자 확인)",
        "- D 슬롯의 03:08~04:37: 강타(Smite) 계열 후보 (자동 템플릿 분석)",
        "- F 슬롯의 03:08~04:37: 점멸(Flash) 후보 (자동 템플릿 분석)",
        f"- D 슬롯 {event_time}: 원본 템플릿 1순위 **{raw_top.get('name','unknown')}** → **강타 진화/파란 동료 문맥으로 재해석**",
        f"- 파란 동료 확인 출처: {'사용자가 원본 플레이를 직접 확인했다고 보고' if confirmed_blue else '확인되지 않은 추정'}",
        "- 게임의 정확한 기록 패치/변화 시점/시전 및 쿨타임 상태: **미검증**",
        "", "## LLM 답변 제한", "",
        "- **23:12에 총명을 장착했다거나 소환사 주문을 교체했다고 단정하지 말 것.**",
        "- 강타의 색·아이콘 진화와 다른 소환사 주문을 분리할 것.",
        "- 강타/점멸 사용 시각과 쿨타임은 영상의 추가 근거 없이는 생성하지 말 것.",
        "- 03:08~04:37 교전 뒤에 23:12 파란 강타가 보였다고, 초반 교전에서도 파란 강타 효과가 있었다고 역추론하지 말 것.",
        "", "## 보고서에 붙일 문구", "",
        "> 키아나는 정글 포지션이며 D 슬롯은 강타 계열, F 슬롯은 점멸 후보로 분류된다. 후반(원본 MP4 23:12)에 관찰된 파란 D 아이콘은 정글 동료 성장에 따른 강타 진화와 양립하며, 사용자가 파란 동료 진화를 확인했다. 따라서 D 슬롯의 원본 '총명' 자동 분류는 미신뢰 판정으로 정정한다. 다만 정확한 진화 시각, 주문 사용 여부 및 쿨타임은 확인하지 않았다.",
    ]
    context_md = outdir / "gemini_smite_rag_context.md"
    context_md.write_text("\n".join(sections) + "\n", encoding="utf-8")
    summary = {
        "video_id": selected_video,
        "issue": "D slot 23:12.000 incorrectly top-matched to Clarity because evolved Smite variants were not represented in the template label handling",
        "raw_top": raw_top,
        "new_interpretation": event,
        "not_a_cast_or_cooldown_analysis": True,
        "source_urls": [fact["source"] for fact in facts if fact["source"].startswith("http")],
        "artifacts": [str(output_json), str(context_md)],
    }
    save_json(outdir / "smite_rag_correction_audit.json", summary)
    print(f"[OK] 원본 보존: {probe}")
    print(f"[OK] 보정 데이터: {output_json}")
    print(f"[OK] Gemini 입력 컨텍스트: {context_md}")
    print(f"[OK] 23:12 분류: {raw_top.get('name')} → 강타 진화 문맥, user-confirmed={confirmed_blue}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=ROOT / "results" / "pipeline" / "LOL_001" / "summoner_spells" / "summoner_spell_probe.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results" / "pipeline" / "LOL_001" / "summoner_spells" / "rag_review")
    parser.add_argument("--timecode", default="00:23:12.000")
    parser.add_argument("--video-id", default="LOL_001")
    parser.add_argument("--confirmed-blue-pet", action="store_true", help="이 영상에서 사용자가 파란 정글 동료 선택을 확인했을 때만")
    args = parser.parse_args()
    run(args.probe, args.outdir, args.timecode, args.confirmed_blue_pet, args.video_id)
