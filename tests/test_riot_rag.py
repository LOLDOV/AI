"""Offline tests. Run in repo root: python -m unittest discover -s tests -p 'test_riot_rag.py'"""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from RAG import cli as rr
from RIOT import client as rc


class RiotRagTests(unittest.TestCase):
    def test_patch_scoping_requires_matching_patch(self):
        self.assertEqual(rr._choose_patch(["16.23.1", "16.22.1"], "16.22.781.1", "match", None), "16.22.1")
        with self.assertRaises(ValueError):
            rr._choose_patch(["16.23.1"], "16.22.781.1", "match", None)
        self.assertEqual(rr._choose_patch(["16.23.1", "16.22.1"], "16.22.781.1", "latest", None), "16.23.1")
        with self.assertRaises(ValueError):
            rr._choose_patch(["16.23.1", "16.22.1"], "16.22.781.1", "match", "16.23.1")

    def test_champion_mismatch_prevents_cross_game(self):
        match = {"info": {"participants": [{"puuid": "p_test", "participantId": 4, "championName": "Qiyana"}]}}
        with self.assertRaises(RuntimeError):
            rr._pick_player(match, "p_test", "Kaisa")
        with self.assertRaises(RuntimeError):
            rr._pick_player(match, "wrong", "Qiyana")
        self.assertEqual(rr._pick_player(match, "p_test", "Qiyana")["participantId"], 4)

    def test_timeline_only_includes_player_or_global_objective(self):
        timeline = {"info": {"frames": [{"events": [
            {"type": "CHAMPION_KILL", "timestamp": 10000, "killerId": 2, "victimId": 3, "assistingParticipantIds": [4]},
            {"type": "CHAMPION_KILL", "timestamp": 11000, "killerId": 2, "victimId": 3, "assistingParticipantIds": []},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 12000, "monsterType": "DRAGON", "killerId": 9},
            {"type": "SKILL_CAST", "timestamp": 13000, "participantId": 4},
        ]}]}}
        found = rr._timeline_events(timeline, 4, 9000, 15000)
        self.assertEqual([e["type"] for e in found], ["CHAMPION_KILL", "ELITE_MONSTER_KILL"])

    def test_rag_retrieves_only_relevant_items(self):
        details = {"data": {"Qiyana": {"id": "Qiyana", "name": "키아나",
            "passive": {"name": "패시브", "description": "기본 공격"},
            "spells": [{"name": x, "description": "기술"} for x in ("Q", "W", "E", "R")]}}}
        items = {"data": {"3006": {"name": "장화", "description": "이동 속도"},
                          "3078": {"name": "삼위일체", "description": "공격 피해"}}}
        docs = rr._docs_from_ddragon(details, items, None, "16.23.1", "ko_KR", {"3006"})
        self.assertTrue(any(d["id"] == "item:3006" for d in docs))
        self.assertFalse(any(d["id"] == "item:3078" for d in docs))
        self.assertTrue(all("16.23.1" in d["source_url"] for d in docs))

    def test_integrated_offline_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            discovery_dir = root / "results" / "discovery" / "X_V4"
            discovery_dir.mkdir(parents=True)
            (discovery_dir / "discovery.json").write_text(json.dumps({
                "status": "complete", "video_path": str(root / "videos" / "X.mp4"),
                "video_fingerprint_partial_sha256": "abc123", "selected_scenes": [
                {"scene_id": "scene_001", "start_sec": 100, "end_sec": 120, "why": "킬 관여"}]}
            ), encoding="utf-8")
            (discovery_dir / "scan_spec.json").write_text(json.dumps({"champion": "Qiyana", "role": "jungle"}), encoding="utf-8")
            match = {"metadata": {"matchId": "KR_123456"}, "info": {
                "mapId": 11, "gameVersion": "16.22.9.11",
                "participants": [{"puuid": "me", "participantId": 4, "championName": "Qiyana", "teamPosition": "JUNGLE", "item0": 3006,
                                  "kills": 1, "deaths": 2, "assists": 3}]}}
            timeline = {"metadata": {"matchId": "KR_123456"}, "info": {"frames": [{"events": [
                {"type": "CHAMPION_KILL", "timestamp": 99000, "killerId": 7, "assistingParticipantIds": [4], "victimId": 9},
                {"type": "CHAMPION_KILL", "timestamp": 100000, "killerId": 1, "assistingParticipantIds": [], "victimId": 2}
            ]}]}}
            for kind, obj in (("matches", match), ("timelines", timeline)):
                p = root / "data" / "riot" / kind / "KR_123456.json"
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps(obj), encoding="utf-8")
            # Cached Data Dragon, so no external network requests are needed.
            cache = root / "data" / "riot" / "ddragon_cache"
            assets = {
                "versions.json": ["16.23.1", "16.22.1"],
                "16.22.1/ko_KR/champion.json": {"data": {"Qiyana": {"id": "Qiyana", "name": "키아나"}}},
                "16.22.1/ko_KR/item.json": {"data": {"3006": {"name": "장화", "description": "이동속도"}}},
                "16.22.1/ko_KR/champions/Qiyana.json": {"data": {"Qiyana": {"id": "Qiyana", "name": "키아나", "passive": {"name": "패시브"}, "spells": [{"name": s} for s in ["Q", "W", "E", "R"]]}}}
            }
            for path, obj in assets.items():
                p = cache / path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(obj), encoding="utf-8")
            args = argparse.Namespace(discovery_id="X_V4", champion=None, region="asia",
                no_match=False, match_id="KR_123456", riot_id=None, puuid="me", confirm_same_match=True,
                fetch=False, patch_mode="match", sync_video="01:40", sync_game="01:39",
                ddragon_version=None, locale="ko_KR", refresh_versions=False, top_docs=8, item_id=[], extra_champion=[])
            with patch.object(rr, "ROOT", root), patch.object(rc, "ROOT", root):
                path = rr.build(args)
                data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["data_dragon_version"], "16.22.1")
            self.assertEqual(data["mp4_minus_game_seconds_user_anchor"], 1)
            self.assertEqual(len(data["scenes"][0]["riot_event_records"]), 1)
            self.assertEqual(data["scenes"][0]["riot_event_records"][0]["killerId"], 7)
            self.assertEqual(data["scenes"][0]["riot_event_records"][0]["assistingParticipantIds"], [4])
            self.assertTrue("16.22.1" in json.dumps(data["scenes"][0]["retrieved_patch_documents"]))
            baseline = rr._prompt_for_scene(data, data["scenes"][0], False)
            assisted = rr._prompt_for_scene(data, data["scenes"][0], True)
            self.assertNotIn('"match_id":', baseline)
            self.assertIn('"match_id":', assisted)
            self.assertNotEqual(baseline, assisted)


if __name__ == "__main__":
    unittest.main()
