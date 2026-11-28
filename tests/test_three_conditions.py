"""Offline tests for the fair 3-condition benchmark. No Riot/Gemini calls."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.three_conditions import runner as r


class DummyDocs:
    version = '16.21.1'
    def __init__(self, version):
        self.game_version = version
    def key(self, name):
        return name
    def local_name(self, name):
        return name
    def names(self):
        return {'Qiyana': 'Qiyana', 'Gwen': 'Gwen'}
    def for_death(self, *args, **kwargs):
        return [{'patch_version': '16.21.1', 'text': '키아나 Q 테스트', 'id': 'champion:Qiyana:Q', 'source_url': 'https://test.invalid'}]


class ThreeConditionTests(unittest.TestCase):
    def test_csv_reads_required_metadata_and_detects_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'gt_videos.csv'
            path.write_text('video_id,match_id,champion\nLOL_033,KR_123,Qiyana\n', encoding='utf-8-sig')
            with patch.object(r, 'ROOT', Path(td)):
                rows = r.read_table(path)
            self.assertEqual(rows[0]['video_id'], 'LOL_033')
            self.assertEqual(rows[0]['match_id'], 'KR_123')
            self.assertEqual(rows[0]['video'].name, 'LOL_033.mp4')
            path.write_text('video_id,match_id,champion\nLOL_033,KR_123,Qiyana\nLOL_033,KR_987,Qiyana\n')
            with self.assertRaises(ValueError):r.read_table(path)

    def test_no_verify_does_not_call_verifier_and_shared_generation_is_single(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            video = td / 'LOL_033.mp4'
            video.write_bytes(b'fake video for mocked capture')
            row = {'video_id': 'LOL_033', 'video': video, 'match_id': 'KR_111',
                   'champion': 'Qiyana', 'riot_id': None, 'puuid': None, 'sync_offset_sec': 0.0}
            match = {'metadata': {'matchId': 'KR_111'}, 'info': {'gameVersion':'16.21.1', 'participants':[
                {'participantId': 7, 'championName': 'Qiyana', 'teamId':200, 'teamPosition':'JUNGLE', 'item0':0}]}}
            timeline = {'metadata': {'matchId': 'KR_111'}, 'info':{'frames':[{'timestamp':25000, 'events':[
                {'type':'CHAMPION_KILL','timestamp':25000,'victimId':7,'killerId':4}]}]}}
            args = SimpleNamespace(run_tag='pilot', fetch=False, limit_deaths=None, rescan=False,
                                   allow_manual_sync=True, execute=True, model='fake-model')
            calls = []
            def fake_capture(video, sc, dest, count):
                dest.mkdir(parents=True, exist_ok=True)
                frames = []
                for i in range(count):
                    p = dest / f'frame_{i}.jpg'
                    p.write_bytes(f'{sc["scene_id"]} frame {i}'.encode())
                    frames.append({'id': f'F{i+1}', 'video_sec': sc['start_sec']+i*0.1,
                                   'rel': i*0.1, 'file': str(p),
                                   'request':('fake', {'frame_id': f'F{i+1}'}, p)})
                return frames
            def fake_generate(model, champion, sc, frames, records, docs):
                calls.append((bool(records), bool(docs), [f['id'] for f in frames]))
                coaching = {'title': '테스트', 'situation':'상황', 'cause':'원인', 'verdict':'check',
                            'claims':[{'id':'c1','kind':'frame','text':'시각 정보','refs':['F1']},
                                      {'id':'c2','kind':'frame','text':'근거 없음','refs':[]}],
                            'alternatives':[], 'checks':[]}
                reply = {'latency_sec': 4, 'token_usage': {'total_tokens':100}, 'raw_text':'{}'}
                return coaching,reply
            def fakedocs(death, match, timeline, player, local_name):
                return {'R1':'death confirmed'}
            with patch.object(r, 'video_meta', return_value={'duration_sec':40}), \
                 patch.object(r.riot_facts, 'load', return_value=(match,timeline)), \
                 patch.object(r, 'PatchDocs', DummyDocs), \
                 patch.object(r.coach, 'capture', side_effect=fake_capture), \
                 patch.object(r.coach, 'generate', side_effect=fake_generate), \
                 patch.object(r.coach, 'record_lines', side_effect=fakedocs):
                first = r.run_one(row,args,out_root=td/'output')
                self.assertEqual(len(calls), 2, 'Full + No Verify must share ONE Gemini generation')
                folder = td/'output'/'pilot'/'LOL_033'
                full = json.loads((folder/'full'/'death_01.json').read_text(encoding='utf-8'))
                no_verify = json.loads((folder/'no_verify'/'death_01.json').read_text(encoding='utf-8'))
                vlm_only = json.loads((folder/'vlm_only'/'death_01.json').read_text(encoding='utf-8'))
                self.assertEqual([x['sha256'] for x in full['frames']], [x['sha256'] for x in no_verify['frames']])
                self.assertEqual([x['sha256'] for x in full['frames']], [x['sha256'] for x in vlm_only['frames']])
                self.assertEqual(len(full['coaching']['claims']), 1)
                self.assertEqual(len(no_verify['coaching']['claims']), 2)
                self.assertEqual(len(vlm_only['coaching']['claims']), 2)
                self.assertEqual(full['check']['applied'], True)
                self.assertEqual(no_verify['check']['applied'], False)
                self.assertEqual(no_verify['record_count_provided'], 1)
                self.assertEqual(vlm_only['record_count_provided'], 0)
                self.assertEqual(vlm_only['patch_doc_count_provided'], 0)
                self.assertEqual(first['errors'], [])
                r.run_one(row,args,out_root=td/'output')
                self.assertEqual(len(calls), 2, 'Repeat run must reuse cache')
                with patch.object(r, 'OUT_BASE', td/'output'):
                    r.sheet(SimpleNamespace(run_tag='pilot', seed=7, allow_partial=False))
                    review = folder.parent/'human_review'
                    # This path is root/output/pilot/human_review, not video-specific.
                    self.assertTrue(review.joinpath('coaching_rating.csv').exists())
                    with review.joinpath('coaching_rating.csv').open(encoding='utf-8-sig') as f:
                        self.assertEqual(len(list(csv.DictReader(f))),3)
                    scores = r.score(SimpleNamespace(run_tag='pilot', ratings=None, claims=None))
                    self.assertEqual(len(scores),3)
                    self.assertEqual(scores[0]['items_rated'],0)
                    rating_file = review / 'coaching_rating.csv'
                    claim_file = review / 'claim_review.csv'
                    with rating_file.open(encoding='utf-8-sig', newline='') as f:
                        ratings = list(csv.DictReader(f))
                    for rating in ratings:
                        for col in r.RUBRIC:
                            rating[col] = '2'
                    with rating_file.open('w', encoding='utf-8-sig', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=list(ratings[0])); w.writeheader(); w.writerows(ratings)
                    with claim_file.open(encoding='utf-8-sig', newline='') as f:
                        claims = list(csv.DictReader(f))
                    for claim in claims:
                        claim['human_label'] = 'SUPPORTED' if claim['claim_no'] == '1' else 'CONTRADICTED'
                    with claim_file.open('w', encoding='utf-8-sig', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=list(claims[0])); w.writeheader(); w.writerows(claims)
                    scored = r.score(SimpleNamespace(run_tag='pilot', ratings=None, claims=None))
                    self.assertEqual(scored[0]['displayed_explicit_error_rate_verified'], 0.0)
                    self.assertEqual(scored[1]['displayed_explicit_error_rate_verified'], 0.5)
                    self.assertEqual(scored[0]['correct_claims_held_by_verifier'], 0)
                    self.assertTrue(all(x['coaching_rating_complete'] for x in scored))

    def test_manual_offset_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            row = {'video_id':'LOL_033', 'video':Path(td)/'test.mp4', 'sync_offset_sec':1.0}
            with self.assertRaisesRegex(ValueError, '--allow-manual-sync'):
                r._get_sync(row,Path(td),rescan=False,allow_manual=False)

    def test_unverified_ref_is_not_proof(self):
        from link import verify
        result=verify.check_claim({'kind':'frame','text':'키아나가 R로 더블킬','refs':['F1']},
             {'frames':{'F1'},'records':{},'docs':{},'patch':'16.21','names':{},'match_champions':set(),'killer':None})
        self.assertEqual(result['status'],'supported')
        # Document this dangerous behavior: human review remains required.

if __name__ == '__main__':
    unittest.main()
