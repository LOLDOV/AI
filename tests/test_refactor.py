import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pipeline import analyze, core, evidence, experiment

class TestRefactor(unittest.TestCase):
    def test_spell_probe_scoped_to_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p=root/'results'/'pipeline'/'TEST'/'summoner_spells'/'summoner_spell_probe.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'video_id':'TEST','slots':{'D':{
                'status':'human_review_required','tentative_name':'강타',
                'template_vote_support':.9,'human_verification':'SECRET_HUMAN_LABEL',
            },'F':{'status':'human_review_required','tentative_name':'점멸',
                   'template_vote_support':1.0}},
                   'user_confirmed_blue_pet':'DO_NOT_INJECT'}), encoding='utf8')
            with patch.object(evidence,'ROOT',root):
                obj=evidence.get_spells('TEST')
            self.assertEqual(obj['slots']['D']['name_candidate'],'강타')
            self.assertNotIn('SECRET_HUMAN_LABEL',json.dumps(obj,ensure_ascii=False))
            self.assertNotIn('DO_NOT_INJECT',json.dumps(obj,ensure_ascii=False))
            self.assertEqual(evidence.get_spells('NON_EXISTENT_TEST')['status'],'not_available')

    def test_analyze_uses_bundled_spells_not_opencv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base=root/'results'/'pipeline'/'UNIT'
            base.mkdir(parents=True)
            core.dump(base/'manifest.json',{'video_id':'UNIT','hud_profile':'configs/hud_1282x832.json',
                      'video_path':'/tmp/video.mp4','scenes':[]})
            (root/'pipeline').mkdir()
            (root/'pipeline'/'spells.py').write_text('')
            with patch.object(analyze,'ROOT',root),\
                 patch.object(analyze,'compile_evidence',return_value={'scenes':[]}),\
                 patch('subprocess.run') as fake_run:
                analyze.analyze('UNIT',spells=True)
                cmd=fake_run.call_args.args[0]
                self.assertTrue(str(root/'pipeline'/'spells.py') in cmd)
                self.assertIn('--scene-root',cmd)
                self.assertIn('--outdir',cmd)

    def test_no_overwrite_or_force_in_experiment(self):
        with self.assertRaises(ValueError):
            experiment.run('TEST','some_model','ab01',force=True)

    def test_manual_cc_not_in_assisted_prompt(self):
        manifest={'video_id':'TEST','champion':'unknown','role':'jungle','scenes':[{
            'scene_id':'scene_001','start_sec':0,'end_sec':1,
            'frames':[{'frame_id':'frame_000.jpg','source_timecode':'00:00:00.000'}]}]}
        ev={'scenes':[{'cc_visual':{'events':[{'status':'visual_CC_candidate_UNCONFIRMED'}]}}]}
        texts=experiment.prompts(manifest,ev)
        self.assertIn('visual_CC_candidate_UNCONFIRMED',texts['assisted'])
        self.assertNotIn('visual_CC_candidate_UNCONFIRMED',texts['baseline'])

    def test_new_video_id_validation(self):
        self.assertEqual(core.safe_id('LOL_002'),'LOL_002')
        for s in ('../../video','a/b','a\\b','..hidden'):
            with self.assertRaises(ValueError):core.safe_id(s)

if __name__=='__main__':unittest.main()
