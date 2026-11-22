import json
from pathlib import Path
import tempfile
import unittest

from pipeline.core import seconds,timecode,parse_scene,safe_id
from pipeline.evidence import get_cc_evidence, get_skill_evidence
from pipeline.experiment import prompts


class TestCore(unittest.TestCase):
    def test_times(self):
        self.assertEqual(seconds('03:08'),188)
        self.assertEqual(seconds('00:03:15.500'),195.5)
        self.assertEqual(timecode(195.5),'00:03:15.500')
        with self.assertRaises(ValueError): seconds('03:90')

    def test_id_and_scene(self):
        scene=parse_scene('qiyana_candidate_01_scene_02=04:15-04:36.967',1)
        self.assertEqual(scene['start_sec'],255)
        self.assertEqual(scene['end_sec'],276.967)
        with self.assertRaises(ValueError):safe_id('../../LOL_001')

    def test_cc_human_labels_not_leaked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            data={'candidate_events':[{'timecode':'00:04:24.000','darkened_slots':['Q','E'],
                 'candidate_pattern':'temporary_dark','human_verified_cc_frame_timecodes':['00:04:24.000'],
                 'cc_confirmed':True,'human_verified_cc_in_visual_window':True}],
                 'human_verified_cc_frames':21,
                 'frames':[{'human_cc_status':'stunned','cc_source':'human_annotation'}]}
            (p/'hud_cc_context.json').write_text(json.dumps(data),encoding='utf-8')
            auto=get_cc_evidence(p)
            dumped=json.dumps(auto)
            self.assertNotIn('stunned',dumped)
            self.assertNotIn('human_verified_cc_',dumped)
            self.assertNotIn('cc_confirmed',dumped)
            self.assertEqual(auto['events'][0]['status'],'visual_CC_candidate_UNCONFIRMED')

    def test_ability_human_labels_not_leaked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            abilities={k:{'visual_state_candidate':'colored_icon_candidate',
                          'verified_by_human':True,'human_label':'ready'} for k in 'QWER'}
            (p/'hud_state_candidates.json').write_text(json.dumps({'frames':[{'abilities':abilities}]}),encoding='utf-8')
            auto=get_skill_evidence(p)
            dumped=json.dumps(auto)
            self.assertNotIn('verified_by_human',dumped)
            self.assertNotIn('human_label',dumped)

    def test_baseline_has_no_auxiliary_evidence(self):
        m={'video_id':'TEST','champion':'Qiyana','role':'jungle',
           'scenes':[{'scene_id':'scene_001','start_sec':0,'end_sec':5,
                     'frames':[{'frame_id':'frame_000.jpg','source_timecode':'00:00:00.000'}]}]}
        ev={'super_secret_marker':'ONLY_MACHINE_EVIDENCE'}
        p=prompts(m,ev)
        self.assertNotIn('ONLY_MACHINE_EVIDENCE',p['baseline'])
        self.assertIn('ONLY_MACHINE_EVIDENCE',p['assisted'])
        self.assertIn('scene_001/frame_000.jpg',p['baseline'])
        self.assertIn('scene_001/frame_000.jpg',p['assisted'])


if __name__=='__main__':unittest.main()
