import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from RIOT.match_resolver import validate_selected_match, match_card
from RIOT.time_sync import consensus_offset, parse_clock_strings
from RAG.patch_rag import _safe_item_snapshot

PUUID = 'test-user-puuid'
MATCH = {'metadata': {'matchId':'KR_1234'},'info':{'mapId':11,'queueId':420,
   'gameDuration':100,'gameVersion':'16.21.1','participants':[
       {'puuid':PUUID,'participantId':7,'championName':'Qiyana','teamPosition':'JUNGLE',
        'kills':9,'deaths':3,'assists':6,'win':True},
       {'puuid':'other','participantId':1,'championName':'Amumu'},
   ]}}


class RiotServiceCoreTests(unittest.TestCase):
    def test_match_ownership_and_duration(self):
        card=validate_selected_match(MATCH,PUUID,'KR_1234',100)
        self.assertEqual(card['champion'],'Qiyana')
        with self.assertRaises(ValueError):
            validate_selected_match(MATCH,PUUID,'KR_1234',500)
        with self.assertRaises(ValueError):
            validate_selected_match(MATCH,PUUID,'KR_9999',100)
        with self.assertRaises(ValueError):
            validate_selected_match(MATCH,'other-account','KR_1234',100)

    def test_disallowed_match_type(self):
        other={'metadata': {'matchId':'KR_1234'}, 'info': {**MATCH['info'], 'mapId':12}}
        self.assertIsNone(match_card(other,PUUID))

    def test_sync_requires_consistent_independent_samples(self):
        pairs=[{'roi':'upper','video_sec':v,'game_sec':v-1} for v in [90,230,450,880]]
        estimated=consensus_offset(pairs)
        self.assertEqual(estimated['offset_sec'],1.0)
        self.assertEqual(estimated['readings'],4)
        wrong=consensus_offset([{'roi':'upper','video_sec':10,'game_sec':1}])
        self.assertEqual(wrong['status'],'unavailable')
        mixed=consensus_offset([{'roi':'a','video_sec':i*100,'game_sec':i*30} for i in range(4)])
        self.assertEqual(mixed['status'],'unavailable')

    def test_clock_parsing(self):
        self.assertEqual(parse_clock_strings('10:39'),[639])
        self.assertEqual(parse_clock_strings('00:10:45'),[645])
        self.assertEqual(parse_clock_strings('nonsense'),[])

    def test_no_bogus_item_description(self):
        item={'name':'요새파괴자','plaintext':'공격력과 관통력',
              'description':'피해 0, 지속시간 25000초',
              'stats':{'FlatPhysicalDamageMod':55},'gold':{'total':3000}}
        safe=_safe_item_snapshot(item)
        self.assertIn('공격력 55',safe)
        self.assertNotIn('25000초',safe)
        self.assertNotIn('피해 0,',safe)

    def test_storage_login_and_scoped_jobs(self):
        from service import storage
        with tempfile.TemporaryDirectory() as d:
            with patch.object(storage,'DB',Path(d)/'service.sqlite'):
                storage.initialize()
                uid=storage.create_user('demo_account','safe_password_123')
                self.assertEqual(storage.check_password('demo_account','safe_password_123'),uid)
                self.assertIsNone(storage.check_password('demo_account','bad_password'))
                token=storage.create_session(uid)
                self.assertEqual(storage.get_user(token)['id'],uid)
                job=storage.create_job(uid,'KR_1234',Path('videos/sample.mp4'))
                self.assertIsNone(storage.get_job(uid+10,job['id']))
                self.assertEqual(storage.get_job(uid,job['id'])['status'],'queued')
                storage.logout(token)
                self.assertIsNone(storage.get_user(token))

    def test_fastapi_login_match_and_upload_without_real_api(self):
        from fastapi.testclient import TestClient
        from service.app import app
        from service import storage
        import cv2
        import numpy as np
        import service.app as app_module
        with tempfile.TemporaryDirectory() as d:
            with patch.object(storage,'DB',Path(d)/'test.sqlite'):
                with TestClient(app) as client:
                    reg=client.post('/api/auth/register',json={'username':'demo_user','password':'safe_password_123'})
                    self.assertEqual(reg.status_code,200,reg.text)
                    with patch('service.app.get_puuid',return_value=PUUID):
                        res=client.post('/api/riot/link',json={'riot_id':'demo#KR1','region':'asia'})
                        self.assertEqual(res.status_code,200,res.text)
                    self.assertFalse(res.json()['ownership_verified'])
                    with patch('service.app.list_recent_matches',return_value=[match_card(MATCH,PUUID)]):
                        history=client.get('/api/riot/matches')
                        self.assertEqual(history.status_code,200)
                        self.assertEqual(history.json()['matches'][0]['match_id'],'KR_1234')
                    path=Path(d)/'sample.mp4'
                    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),4,(64,64))
                    self.assertTrue(writer.isOpened())
                    for i in range(400):
                        writer.write(np.zeros((64,64,3),dtype=np.uint8))
                    writer.release()
                    # Patch ROOT for the upload path only: keep all project-specific paths in fake temp dir.
                    with patch.object(app_module,'ROOT',Path(d)):
                        with patch('service.app._riot_json',return_value=MATCH),patch('service.app.submit_analysis') as submit:
                            with path.open('rb') as f:
                                response=client.post('/api/analyses',data={'match_id':'KR_1234'},
                                    files={'video':('sample.mp4',f,'video/mp4')})
                            self.assertEqual(response.status_code,200,response.text)
                            self.assertEqual(response.json()['champion'],'Qiyana')
                            self.assertEqual(submit.call_count,1)
                            jid=response.json()['job_id']
                            self.assertEqual(client.get('/api/analyses/'+jid).status_code,200)
                            self.assertEqual(client.get('/api/analyses/invalid').status_code,404)


if __name__=='__main__':
    unittest.main()

# Additional tests in a separate class so this file remains executable via unittest discovery.
class AutoWorkflowTests(unittest.TestCase):
    def test_unavailable_clock_does_not_invent_riot_events(self):
        import json
        from unittest.mock import patch,call
        import service.workflow as workflow
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            did='AUTO_testclock'
            discovery_path=root/'results'/'discovery'/did/'discovery.json'
            discovery_path.parent.mkdir(parents=True,exist_ok=True)
            discovery_path.write_text(json.dumps({'status':'complete','selected_scenes':[{'scene_id':'scene_001'}]}))
            result_path=root/'results'/'discovery'/did/'riot_rag'/'coaching'/'riot_rag_ab01'/'scene_001'/'assisted_result.json'
            result_path.parent.mkdir(parents=True,exist_ok=True)
            result_path.write_text('{}')
            commands=[]
            def fake_call(job_id, argv, log, stage):
                commands.append((stage,argv))
            with patch.object(workflow,'ROOT',root),patch.object(workflow,'LOG_DIR',root/'logs'), \
                patch.object(workflow,'_call',side_effect=fake_call), \
                patch.object(workflow,'estimate_from_video',return_value={'status':'unavailable'}), \
                patch.object(workflow,'update_job') as statuses:
                workflow.run_analysis('jobid',Path(d)/'video.mp4','KR_1234',PUUID,'Qiyana','JUNGLE',did)
                self.assertEqual([x[0] for x in commands],['discover','riot_rag_build','coaching'])
                self.assertNotIn('--sync-video',commands[1][1])
                self.assertNotIn('--sync-game',commands[1][1])
                self.assertIn(call('jobid',sync_status='unavailable_no_timeline_events'),statuses.mock_calls)
                self.assertIn(call('jobid',status='complete',stage='done'),statuses.mock_calls)
