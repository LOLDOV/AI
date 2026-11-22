import unittest

from pipeline.combat_discovery import (
    build_windows,normalize_response,stratified_shortlist,_rank_verified,
    scout_candidates,
)

class TestDiscover(unittest.TestCase):
    def test_full_duration_including_end(self):
        windows=build_windows(1500,24,4)
        self.assertEqual(len(windows),63)
        self.assertEqual(windows[0]['start_sec'],0)
        self.assertEqual(windows[-1]['end_sec'],1500)
        self.assertTrue(any(w['start_sec']>=1400 for w in windows))
        self.assertEqual(len({w['window_id'] for w in windows}),len(windows))

    def test_respawn_and_farming_are_not_selected(self):
        windows=build_windows(100,24,4)
        obj={'windows':[
            {'window_id':'w0001','category':'death_respawn','importance':5},
            {'window_id':'w0002','category':'jungle_farming','importance':5},
            {'window_id':'w0003','category':'champion_fight','importance':4},
            {'window_id':'w0004','category':'lane_trade','importance':4},
            {'window_id':'w0005','category':'objective_fight','importance':4},
        ]}
        classified=normalize_response(obj,windows)
        self.assertEqual([r['window_id'] for r in scout_candidates(classified)],['w0003','w0005'])
        self.assertEqual(classified[0]['score'],0)
        self.assertEqual(classified[1]['score'],0)

    def test_missing_model_classification_is_unknown_not_negative(self):
        windows=build_windows(50,24,4)
        rows=normalize_response({'windows':[{'window_id':'w0001','category':'champion_fight',
                'importance':3,'evidence_images':['w0001_f01','fabricated_image']}]},windows)
        self.assertEqual(rows[0]['evidence_images'],['w0001_f01'])
        self.assertTrue(rows[1]['missing_response'])
        self.assertEqual(rows[1]['category'],'unknown')

    def test_global_ranking_not_first_five(self):
        windows=build_windows(1000,24,4)
        rowmap={w['window_id']:w for w in windows}
        # Five poor early candidates should never outrank strong fights late in VOD.
        weak=[{'window_id':w['window_id'],'category':'possible_champion_fight','importance':2,'score':5}
              for w in windows[:5]]
        late=[{'window_id':windows[-i]['window_id'],'category':'champion_fight','importance':4,'score':11}
              for i in (1,2)]
        picks=stratified_shortlist(weak+late,rowmap,limit=7,duration=1000)
        self.assertTrue(any(r['window_id']==windows[-1]['window_id'] for r in picks))

    def test_dense_downgrade_to_movement_rejects(self):
        windows=build_windows(96,24,4)
        scout=[{'window_id':'w0001','category':'champion_fight','importance':5,'score':12},
               {'window_id':'w0003','category':'possible_champion_fight','importance':3,'score':6}]
        verified={'w0001':{'window_id':'w0001','category':'movement','importance':0,'score':0,'why':'walk'},
                  'w0003':{'window_id':'w0003','category':'champion_fight','importance':4,'score':11,'why':'fight'}}
        selected,all_c=_rank_verified(windows,scout,verified,top_k=5)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['window_id'],'w0003')
        self.assertEqual(selected[0]['origin'],'vlm_storyboard_and_dense_candidate_UNVERIFIED')

if __name__=='__main__': unittest.main()

class TestMergeCombat(unittest.TestCase):
    def test_adjacent_fight_windows_are_merged(self):
        windows=build_windows(120,24,4)
        sc=[{'window_id':'w0002','category':'champion_fight','importance':3,'score':10},
            {'window_id':'w0003','category':'champion_fight','importance':5,'score':12}]
        vr={wid:{'window_id':wid,'category':'champion_fight','importance':3,'score':10,'why':'fight'}
            for wid in ('w0002','w0003')}
        picks,_=_rank_verified(windows,sc,vr,top_k=5)
        self.assertEqual(len(picks),1)
        self.assertEqual(picks[0]['source_window_ids'],['w0002','w0003'])
        self.assertEqual(picks[0]['start_sec'],19)
        self.assertEqual(picks[0]['end_sec'],80)
