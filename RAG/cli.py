"""Top-level Riot + patch RAG integration CLI. No new code in pipeline/.

Examples:
  python -m RAG.cli build --discovery-id LOL_002_COACH_V4 --no-match --champion Kaisa --patch-mode latest
  python -m RAG.cli build --discovery-id LOL_002_COACH_V4 --match-id KR_123456 \
      --riot-id 'name#KR1' --confirm-same-match --sync-video 11:40 --sync-game 11:38
  python -m RAG.cli coach --discovery-id LOL_002_COACH_V4 --ab
"""
from __future__ import annotations
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from pipeline.core import ROOT, dump, load, safe_id
from RIOT.client import (
    RIOT_REGIONS, _get, _cache_json, _load_env, _riot_json,
    _puuid_from_riot_id, _parse_time, _pick_player, _timeline_events,
)
from RAG.patch_rag import (
    DD_BASE, DOCS_SOURCE, _choose_patch, _docs_from_ddragon, _retrieve,
)
from RAG.coaching import coach, _prompt_for_scene


def _scene_records(discovery: dict) -> list[dict]:
    scenes = discovery.get("selected_scenes", [])
    if not scenes:
        raise RuntimeError("discovery.json의 selected_scenes가 비어 있습니다")
    for scene in scenes:
        if "scene_id" not in scene or "start_sec" not in scene or "end_sec" not in scene:
            raise RuntimeError("selected_scenes 구조가 예상과 다릅니다")
    return scenes


def build(args) -> Path:
    discovery_id = safe_id(args.discovery_id)
    root = ROOT / "results" / "discovery" / discovery_id
    discovery_file = root / "discovery.json"
    if not discovery_file.exists():
        raise FileNotFoundError(f"먼저 v4 discover를 끝내야 함: {discovery_file}")
    discovery = load(discovery_file)
    if discovery.get("status") != "complete":
        raise RuntimeError("완료된 discovery.json이 아닙니다")
    scenes = _scene_records(discovery)
    spec_file = root / "scan_spec.json"
    spec = load(spec_file) if spec_file.exists() else {}
    champion = args.champion or spec.get("champion") or "unknown"
    region = args.region.lower()
    if region not in RIOT_REGIONS:
        raise ValueError("--region: asia, americas, europe, sea 중 선택")

    if args.no_match:
        if args.match_id or args.riot_id or args.puuid or args.confirm_same_match:
            raise ValueError("--no-match는 Riot 경기 연결 옵션과 동시에 사용할 수 없습니다")
        if args.patch_mode != "latest":
            raise ValueError("경기 미연결 모드는 --patch-mode latest로만 가능")
        if champion == "unknown":
            raise ValueError("--champion 값을 지정하세요")
        match = None
        timeline = None
        player = None
        actual_match_id = None
        game_version = "unknown"
        provenance = "NO_MATCH_PATCH_ONLY_NOT_GAME_PROOF"
    else:
        if not args.match_id or not args.confirm_same_match:
            raise ValueError("경기 연결에는 --match-id 및 --confirm-same-match가 필수입니다")
        if not args.riot_id and not args.puuid:
            raise ValueError("플레이어 확인에는 --riot-id 또는 --puuid가 필수입니다")
        if args.patch_mode == "latest":
            raise ValueError("경기 연결에서는 --patch-mode latest 금지. 해당 경기의 패치를 사용해야 합니다")
        actual_match_id = args.match_id.upper()
        match = _riot_json(actual_match_id, "match", fetch=args.fetch, region=region)
        timeline = _riot_json(actual_match_id, "timeline", fetch=args.fetch, region=region)
        if match.get("metadata", {}).get("matchId", "").upper() != actual_match_id:
            raise RuntimeError("Match JSON의 matchId가 입력과 달라 연결할 수 없습니다")
        if timeline.get("metadata", {}).get("matchId", "").upper() != actual_match_id:
            raise RuntimeError("Timeline JSON의 matchId가 Match JSON과 다릅니다")
        puuid = args.puuid or _puuid_from_riot_id(args.riot_id, region=region)
        player = _pick_player(match, puuid, champion)
        champion = player["championName"]
        game_version = str(match.get("info", {}).get("gameVersion") or "")
        provenance = "USER_CONFIRMED_SAME_MATCH_NOT_AUTOMATICALLY_PROVEN"
        if match.get("info", {}).get("mapId") != 11:
            raise RuntimeError("소환사의 협곡(mapId=11) 경기가 아닙니다")

    if bool(args.sync_video) != bool(args.sync_game):
        raise ValueError("--sync-video / --sync-game은 반드시 함께 제공하세요")
    if args.no_match and args.sync_video:
        raise ValueError("Riot 경기 연결 없이 시간 동기화할 수 없습니다")
    offset_sec = _parse_time(args.sync_video) - _parse_time(args.sync_game) if args.sync_video else None

    cache = ROOT / "data" / "riot" / "ddragon_cache"
    versions_path = cache / "versions.json"
    version_list_stale = (versions_path.exists() and args.patch_mode == "latest" and
                          time.time() - versions_path.stat().st_mtime > 12 * 3600)
    versions = _cache_json(versions_path, f"{DD_BASE}/api/versions.json",
                           refresh=args.refresh_versions or version_list_stale)
    version = _choose_patch(versions, game_version, args.patch_mode, args.ddragon_version)
    base_url = f"{DD_BASE}/cdn/{version}/data/{args.locale}"
    all_champs = _cache_json(cache / version / args.locale / "champion.json", base_url + "/champion.json")
    items = _cache_json(cache / version / args.locale / "item.json", base_url + "/item.json")
    candidates = all_champs.get("data", {})
    # canonical champion key (e.g. MonkeyKing vs Wukong) and Korean localized names.
    champion_key = next((key for key, row in candidates.items()
                         if key.casefold() == str(champion).casefold() or
                         str(row.get("name", "")).casefold() == str(champion).casefold()), None)
    if not champion_key:
        raise RuntimeError(f"Data Dragon {version}에서 챔피언을 찾지 못함: {champion}")
    champ_url = base_url + f"/champion/{quote(champion_key)}.json"
    details = _cache_json(cache / version / args.locale / "champions" / f"{champion_key}.json", champ_url)
    chosen_items = set(args.item_id or [])
    if timeline and player:
        pid = int(player["participantId"])
        for frame in timeline.get("info", {}).get("frames", []):
            for ev in frame.get("events", []):
                if ev.get("type") == "ITEM_PURCHASED" and ev.get("participantId") == pid:
                    if isinstance(ev.get("itemId"), int) and ev["itemId"] > 0:
                        chosen_items.add(str(ev["itemId"]))
    docs = _docs_from_ddragon(details, items, player, version, args.locale, chosen_items)
    extra_champion_sources = []
    extras = list(dict.fromkeys(args.extra_champion or []))
    if len(extras) > 5:
        raise ValueError("--extra-champion: 최대 5명")
    known_match_champions = ({p.get("championName", "").casefold()
                             for p in match.get("info", {}).get("participants", [])}
                            if match else None)
    for extra in extras:
        other_key = next((key for key, row in candidates.items()
                          if key.casefold() == extra.casefold() or
                          str(row.get("name", "")).casefold() == extra.casefold()), None)
        if other_key is None:
            raise ValueError(f"Data Dragon에 없는 챔피언: {extra}")
        if match and other_key.casefold() not in known_match_champions:
            raise ValueError(f"이 경기 참가자가 아닌 챔피언: {extra}")
        if other_key == champion_key:
            continue
        other_url = base_url + f"/champion/{quote(other_key)}.json"
        other_details = _cache_json(cache / version / args.locale / "champions" / f"{other_key}.json",
                                    other_url)
        docs.extend(_docs_from_ddragon(other_details, items, None, version, args.locale, set()))
        extra_champion_sources.append(other_url)

    enriched = []
    for scene in scenes:
        # frame MP4 time -> game time; no pairing when anchor unavailable.
        start_game_sec = scene["start_sec"] - offset_sec if offset_sec is not None else None
        end_game_sec = scene["end_sec"] - offset_sec if offset_sec is not None else None
        events = []
        if timeline and start_game_sec is not None:
            events = _timeline_events(timeline, int(player["participantId"]),
                                      round(max(0, start_game_sec) * 1000), round(max(0, end_game_sec) * 1000))
        query = f"{champion} {scene.get('why', '')} {' '.join(scene.get('coaching_signals',[]))}"
        enriched.append({"scene_id": scene["scene_id"], "start_mp4_sec": scene["start_sec"],
                         "end_mp4_sec": scene["end_sec"], "vlm_unverified_summary": scene.get("why"),
                         "timeline_alignment": ("user_anchor_estimate_not_independently_verified"
                                                if offset_sec is not None else "unavailable_no_clock_anchor"),
                         "game_time_range_sec_estimate": ([round(start_game_sec, 3), round(end_game_sec, 3)]
                                                           if offset_sec is not None else None),
                         "riot_event_records": events,
                         "retrieved_patch_documents": _retrieve(docs, query, limit=args.top_docs),
                         "restrictions": ["Riot timeline has no skill-cast/dodge or enemy HP event log",
                                          "A Data Dragon item in the final build is NOT proof of owning it during this scene",
                                          "VLM summary is an unverified hypothesis, not a factual label"]})
    video_path = discovery.get("video_path")
    data = {
        "schema": "loldov-riot-patch-rag-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_id": discovery_id,
        "video_path": video_path,
        "source_discovery_fingerprint": discovery.get("video_fingerprint_partial_sha256"),
        "game_match_identity_status": provenance,
        "match_id": actual_match_id,
        "champion": champion,
        "role_user_declared": spec.get("role", "unknown"),
        "match_game_version": game_version,
        "data_dragon_version": version,
        "patch_mode": args.patch_mode,
        "patch_data_is_not_patch_notes": True,
        "riot_source": DOCS_SOURCE,
        "data_dragon_sources": [base_url + "/champion.json", champ_url,
                                base_url + "/item.json", *extra_champion_sources],
        "player_match_summary": ({"participantId": player.get("participantId"),
                                  "championName": player.get("championName"),
                                  "teamPosition": player.get("teamPosition"),
                                  "kills": player.get("kills"), "deaths": player.get("deaths"),
                                  "assists": player.get("assists"), "win": player.get("win"),
                                  "final_items_only": [player.get(f"item{n}") for n in range(7)]}
                                 if player else None),
        "mp4_minus_game_seconds_user_anchor": round(offset_sec, 3) if offset_sec is not None else None,
        "scenes": enriched,
        "warnings": ["USER confirmation of matching VOD and Match ID is NOT automated video verification",
                     "No anchor => timeline events are not attached to scenes",
                     "Selected VLM candidates may include incorrect facts",
                     "Patch Data Dragon describes champion/item snapshot, NOT official patch notes or balance changes",
                     "The latest patch is not necessarily the patch in which the VOD was recorded"],
    }
    output = root / "riot_rag"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "context_bundle.json"
    if path.exists():
        previous = load(path)
        important = ("match_id", "data_dragon_version", "mp4_minus_game_seconds_user_anchor", "champion")
        if any(previous.get(k) != data.get(k) for k in important):
            raise FileExistsError("기존 context_bundle과 입력이 다릅니다. 원본 보존 위해 --discovery-id를 새로 복사해 사용하거나 riot_rag 폴더 별도 백업 후 변경하세요")
    dump(path, data)
    print("[OK] context:", path)
    print("[PATCH]", version, "game:", game_version, "mode:", args.patch_mode)
    print("[LINK]", provenance, "/ anchor offset:", offset_sec)
    print("[SCENES]", len(enriched), " / Riot linked events:", sum(len(s["riot_event_records"]) for s in enriched))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m RAG.cli", description="Riot and patch RAG AFTER V4 discovery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Create patch-scoped data/Riot event evidence per selected scene")
    b.add_argument("--discovery-id", required=True)
    b.add_argument("--match-id", help="Exact match ID linked to VOD (required unless --no-match)")
    b.add_argument("--riot-id", help="Player Riot ID, e.g. 'name#KR1' (never share credentials)")
    b.add_argument("--puuid", help="Alternative to --riot-id")
    b.add_argument("--no-match", action="store_true", help="Only fetch latest patch static data, NO Riot match evidence")
    b.add_argument("--champion", help="Required for --no-match if scan_spec.json has no champion")
    b.add_argument("--item-id", action="append", default=[], help="Optional relevant item ID (repeatable, e.g. --item-id 3006)")
    b.add_argument("--extra-champion", action="append", default=[],
                   help="Optional other champion kit in same match, e.g. --extra-champion Lux")
    b.add_argument("--confirm-same-match", action="store_true", help="User verified recording is this exact match")
    b.add_argument("--sync-video", help="MP4 clock observed at one common action, e.g. 01:15")
    b.add_argument("--sync-game", help="Game HUD clock at the SAME action, e.g. 01:12")
    b.add_argument("--fetch", action="store_true", help="Fetch missing Riot match/timeline JSON using RIOT_API_KEY")
    b.add_argument("--region", default="asia", choices=sorted(RIOT_REGIONS))
    b.add_argument("--patch-mode", choices=["match", "latest"], default="match")
    b.add_argument("--ddragon-version", default=None, help="Exact approved Data Dragon version")
    b.add_argument("--refresh-versions", action="store_true", help="Refresh available versions list")
    b.add_argument("--locale", default="ko_KR", choices=["ko_KR", "en_US"])
    b.add_argument("--top-docs", type=int, default=8)
    c = sub.add_parser("coach", help="Use identical VOD frames for baseline vs Riot+RAG assisted coaching")
    c.add_argument("--discovery-id", required=True)
    c.add_argument("--video", help="Optional video override, fingerprint must be identical")
    c.add_argument("--model", default=None)
    c.add_argument("--run-tag", default="riot_rag_ab01")
    c.add_argument("--frames-per-scene", type=int, default=6)
    c.add_argument("--ab", action="store_true", help="Compare same frames VLM-only vs Riot+RAG")
    c.add_argument("--execute", action="store_true", help="Real Gemini requests, incurs charges")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        if not (1 <= args.top_docs <= 30):
            ap.error("--top-docs: 1..30")
        build(args)
    else:
        if not (2 <= args.frames_per_scene <= 16):
            ap.error("--frames-per-scene: 2..16")
        coach(args)


if __name__ == "__main__":
    main()
