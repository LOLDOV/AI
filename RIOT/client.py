"""Riot Match/Timeline loading, identity checks, and event mapping.

No discovery or coaching algorithms here. Saved data remains under data/riot/.
"""
from __future__ import annotations
import os
import re
import time
from pathlib import Path
from urllib.parse import quote
import requests
from pipeline.core import ROOT, load, dump

RIOT_REGIONS = {"asia", "americas", "europe", "sea"}


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
    except ImportError:
        pass


def _get(url: str, *, headers: dict | None = None, attempts: int = 3) -> dict | list:
    for n in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=25)
        except requests.RequestException as exc:
            if n == attempts - 1:
                raise RuntimeError(f"Network request failed for {url}: {exc}") from exc
            time.sleep(2 ** n)
            continue
        if response.status_code == 429:
            if n == attempts - 1:
                raise RuntimeError(f"Rate limited (429): {url}")
            delay = response.headers.get("Retry-After", "3")
            try:
                seconds = max(1.0, min(30.0, float(delay)))
            except ValueError:
                seconds = 3.0
            time.sleep(seconds)
            continue
        if response.status_code in {500, 502, 503, 504} and n < attempts - 1:
            time.sleep(2 ** n)
            continue
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} requesting {url}: {response.text[:120]}")
        return response.json()
    raise RuntimeError(f"Request failed: {url}")


def _cache_json(path: Path, url: str, *, headers: dict | None = None, refresh: bool = False):
    if path.exists() and not refresh:
        return load(path)
    value = _get(url, headers=headers)
    dump(path, value)
    return value


def _riot_json(match_id: str, kind: str, *, fetch: bool, region: str):
    if not re.fullmatch(r"[A-Z0-9]+_\d+", match_id, re.I):
        raise ValueError("match-id 형식 오류. 예: KR_1234567890")
    folder = "matches" if kind == "match" else "timelines"
    path = ROOT / "data" / "riot" / folder / f"{match_id}.json"
    if path.exists():
        return load(path)
    if not fetch:
        raise FileNotFoundError(f"Riot {kind} JSON 없음: {path}. 기존 riot_api.py로 수집하거나 --fetch 사용")
    _load_env()
    key = os.getenv("RIOT_API_KEY", "").strip()
    if not key:
        raise RuntimeError(".env의 RIOT_API_KEY가 없습니다. 실제 키는 채팅·GitHub에 올리지 마세요")
    endpoint = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{quote(match_id)}"
    if kind == "timeline":
        endpoint += "/timeline"
    return _cache_json(path, endpoint, headers={"X-Riot-Token": key})


def _puuid_from_riot_id(riot_id: str, *, region: str):
    if "#" not in riot_id:
        raise ValueError("--riot-id 형식은 이름#태그여야 합니다")
    name, tag = riot_id.rsplit("#", 1)
    if not name.strip() or not tag.strip():
        raise ValueError("Riot ID의 이름과 태그가 필요합니다")
    # If existing riot_api.py indexed this exact Riot ID, don't call Riot again.
    index = ROOT / "data" / "riot" / "account_matches.json"
    if index.exists():
        accounts = load(index)
        entry = next((v for k, v in accounts.items() if k.casefold() == riot_id.casefold()), None)
        if isinstance(entry, dict) and entry.get("puuid"):
            return entry["puuid"]
    _load_env()
    key = os.getenv("RIOT_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Riot ID와 PUUID가 로컬 index에 없고 RIOT_API_KEY도 없습니다. 또는 --puuid 사용")
    url = (f"https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/"
           f"{quote(name, safe='')}/{quote(tag, safe='')}")
    return _get(url, headers={"X-Riot-Token": key})["puuid"]


def _parse_time(raw: str) -> float:
    parts = str(raw).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"시각 형식 오류: {raw}. 01:30 또는 01:30:01")
    nums = [float(v) for v in parts]
    if any(v < 0 for v in nums) or (len(nums) > 1 and any(v >= 60 for v in nums[1:])):
        raise ValueError(f"시각 범위 오류: {raw}")
    out = 0.0
    for n in nums:
        out = out * 60 + n
    return out


def _pick_player(match: dict, puuid: str, requested_champion: str):
    participants = match.get("info", {}).get("participants", [])
    matched = [p for p in participants if p.get("puuid") == puuid]
    if len(matched) != 1:
        raise RuntimeError("요청한 플레이어의 PUUID가 경기 participant에 정확히 1명 존재하지 않음. 연결 중단")
    player = matched[0]
    actual_champion = str(player.get("championName") or "")
    if requested_champion and requested_champion.lower() != "unknown" and \
            actual_champion.casefold() != requested_champion.casefold():
        raise RuntimeError(f"챔피언 불일치: VOD {requested_champion}, Match {actual_champion}. 다른 경기일 가능성")
    return player


def _timeline_events(timeline: dict, pid: int, start_ms: int, end_ms: int) -> list[dict]:
    """Only log events involving the player, or global objective kills.
    Riot timeline does NOT provide skill cast, dodge, ward vision or exact HP-at-event.
    """
    result = []
    for frame in timeline.get("info", {}).get("frames", []):
        for ev in frame.get("events", []):
            event_ms = ev.get("timestamp")
            if not isinstance(event_ms, (int, float)) or not start_ms <= event_ms <= end_ms:
                continue
            kind = ev.get("type", "")
            involved = (ev.get("participantId") == pid or ev.get("killerId") == pid
                        or ev.get("victimId") == pid or pid in (ev.get("assistingParticipantIds") or [])
                        or ev.get("creatorId") == pid)
            is_objective = kind in {"ELITE_MONSTER_KILL", "BUILDING_KILL"}
            if not (involved or is_objective):
                continue
            if kind not in {"CHAMPION_KILL", "ELITE_MONSTER_KILL", "BUILDING_KILL", "ITEM_PURCHASED",
                            "ITEM_SOLD", "ITEM_DESTROYED", "ITEM_UNDO", "WARD_PLACED", "WARD_KILL"}:
                continue
            safe_keys = ("type", "timestamp", "killerId", "victimId", "assistingParticipantIds",
                         "participantId", "monsterType", "monsterSubType", "buildingType", "laneType",
                         "itemId", "beforeId", "afterId", "killerTeamId", "wardType", "teamId")
            record = {key: ev[key] for key in safe_keys if key in ev}
            record["player_involvement"] = ("participant_or_assist" if involved
                                            else "global_event_only_NOT_player_contribution")
            result.append(record)
    return sorted(result, key=lambda ev: ev["timestamp"])
