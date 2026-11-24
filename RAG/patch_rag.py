"""Patch-scoped Data Dragon text retrieval (NOT a patch notes crawler)."""
from __future__ import annotations
import html
import re
from urllib.parse import quote
from RIOT.client import _cache_json

DD_BASE = "https://ddragon.leagueoflegends.com"
PATCH_RE = re.compile(r"^(\d+\.\d+)(?:\.|$)")
DOCS_SOURCE = "https://developer.riotgames.com/docs/lol"


def _clean_text(raw: str | None) -> str:
    value = re.sub(r"<[^>]+>", " ", raw or "")
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()[:1100]


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[가-힣A-Za-z0-9]{2,}", value or "")}


def _choose_patch(versions: list, game_version: str, mode: str, override: str | None):
    if override:
        if override not in versions:
            raise ValueError(f"Data Dragon 목록에 없는 버전: {override}")
        if mode == "match" and game_version:
            match = PATCH_RE.match(game_version)
            if not match or not override.startswith(match.group(1) + "."):
                raise ValueError(f"선택 버전 {override}과 실제 경기 gameVersion {game_version}이 일치하지 않습니다")
        return override
    if mode == "latest":
        return versions[0]
    match = PATCH_RE.match(game_version or "")
    if not match:
        raise ValueError("Riot Match JSON의 gameVersion 확인 불가. 최신 버전으로 자동 대체하지 않습니다")
    prefix = match.group(1) + "."
    for version in versions:
        if version.startswith(prefix):
            return version
    raise ValueError(f"경기 패치 {game_version}에 대응하는 Data Dragon 버전 없음. --patch-mode latest로 강제 대체하지 마세요")


def _docs_from_ddragon(champ_details: dict | None, items: dict, player: dict | None,
                       patch: str, locale: str, chosen_items: set[str]):
    docs = []
    base = f"{DD_BASE}/cdn/{patch}/data/{locale}"
    if champ_details:
        champion = next(iter(champ_details["data"].values()))
        cname = champion.get("name", champion.get("id", ""))
        cid = champion.get("id", "")
        source = f"{base}/champion/{quote(cid)}.json"
        docs.append({"id": f"champion:{cid}:passive", "kind": "champion_ability", "subject": cname,
                     "text": f"{cname} 기본 지속 효과: {_clean_text(champion.get('passive', {}).get('name'))}. {_clean_text(champion.get('passive', {}).get('description'))}",
                     "source_url": source, "patch_version": patch})
        for slot, spell in zip("QWER", champion.get("spells", [])[:4]):
            docs.append({"id": f"champion:{cid}:{slot}", "kind": "champion_ability", "subject": cname,
                         "text": f"{cname} {slot}: {_clean_text(spell.get('name'))}. {_clean_text(spell.get('description'))}",
                         "source_url": source, "patch_version": patch})
    # Final loadout is NOT historical scene inventory. Never imply these items were owned then.
    final_ids = set()
    if player:
        for n in range(7):
            item_id = player.get(f"item{n}")
            if isinstance(item_id, int) and item_id > 0:
                final_ids.add(str(item_id))
    allowed = final_ids | chosen_items
    for item_id, item in items.get("data", {}).items():
        if item_id not in allowed:
            continue
        docs.append({"id": f"item:{item_id}", "kind": "item", "subject": item.get("name", item_id),
                     "text": f"아이템 {item.get('name', item_id)}: {_clean_text(item.get('description'))} ",
                     "source_url": f"{base}/item.json", "patch_version": patch,
                     "note": "경기 기록에서 등장한 아이템; 이 장면 당시 보유 여부는 미검증" if player else "패치 데이터 참고용; 실제 경기 사용 여부 미확인"})
    return docs


def _retrieve(docs: list[dict], query: str, limit: int = 8) -> list[dict]:
    terms = _tokens(query)
    ranked = []
    for doc in docs:
        text = f"{doc.get('subject','')} {doc.get('text','')}"
        overlap = len(terms & _tokens(text))
        bonus = 1.0 if doc["kind"] == "champion_ability" else 0
        score = overlap + bonus
        ranked.append((score, doc))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]["id"]))
    return [{**doc, "retrieval_score_not_confidence": score} for score, doc in ranked[:limit]]
