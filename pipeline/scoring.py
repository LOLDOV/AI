"""Summarize HUMAN rubric scores only; not an automatic accuracy metric."""
from __future__ import annotations
import csv
from pathlib import Path
from .core import ROOT,safe_id,dump


def score(video_id:str,run_tag:str):
    path=ROOT/'results'/'pipeline'/safe_id(video_id)/'experiments'/safe_id(run_tag)
    if not (path/'baseline_result.json').exists() or not (path/'assisted_result.json').exists():
        raise FileNotFoundError('Both Gemini results required before scoring.')
    template=path/'comparison_template.csv'
    with template.open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    usable=[];pending=[]
    for r in rows:
        left=str(r['baseline_score_0_2_or_NA']).strip().upper()
        right=str(r['assisted_score_0_2_or_NA']).strip().upper()
        if left in {'0','1','2'} and right in {'0','1','2'}:
            usable.append((r['criterion'],int(left),int(right),r.get('evidence_and_notes','')))
        elif not left or not right or left=='NA' or right=='NA':
            pending.append(r['criterion'])
        else:
            raise ValueError(f'Bad score: {r["criterion"]}: {left!r},{right!r}; use 0,1,2,NA')
    base=sum(r[1] for r in usable);assist=sum(r[2] for r in usable);maxval=2*len(usable)
    result={'schema':'loldov_subjective_rubric_v1','video_id':video_id,'run_tag':run_tag,
            'paired_scored_items':len(usable),'pending_or_not_comparable_items':len(pending),
            'baseline_score':base,'assisted_score':assist,'maximum_comparable_score':maxval,
            'score_delta_assisted_minus_baseline':assist-base,
            'claim':'Rubric scores are human-provided judgments, NOT automated factual accuracy nor statistically validated model performance.',
            'items':[{'criterion':name,'baseline':a,'assisted':b,'notes':note} for name,a,b,note in usable]}
    dump(path/'comparison_scores.json',result)
    md=['# 사람이 평가한 코칭 품질 비교 (정확도 아님)','',
        f'- 동일 기준으로 채점한 항목: **{len(usable)}/{len(rows)}**',
        f'- Baseline: **{base}/{maxval}**',f'- Assisted: **{assist}/{maxval}**',
        f'- 점수 차이: **{assist-base:+d}점**', '',
        '> 주의: 평가자의 주관적 루브릭 점수입니다. 영상 내 사실 정확도/일반화 성능을 증명하지 않습니다.',
        '', '| 항목 | Baseline | Assisted | 근거 |', '|---|---:|---:|---|']
    md += [f'| {name.replace("|","/")} | {a} | {b} | {(note or "-").replace("|","/").replace(chr(10)," ")} |' for name,a,b,note in usable]
    if pending:md+=['','채점 보류 항목: '+', '.join(pending)]
    (path/'comparison_scored.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    return path/'comparison_scored.md'
