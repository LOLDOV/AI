"""Claim-level evidence check for link coaching.

Each claim carries refs (F frames, R Riot records, D patch docs). Status:
  supported    refs exist and match the claim kind
  unsupported  no refs, unknown refs, or the wrong kind of evidence
  conflict     champion not in the match, a killer the record contradicts,
               or a patch number missing from the cited document
  stale        cites a document from a different patch than the match
This checks the link between a claim and its evidence; whether the claim is true is left to human rating.
"""
from __future__ import annotations

import re

from . import riot_facts

KIND_PREFIX = {'frame': 'F', 'record': 'R', 'patch': 'D'}
KILL_RE = re.compile(r'([0-9A-Za-z가-힣]+?)(?:에게|한테|에 의해)\s*(?:처치|죽|잡|킬)')


def build_context(frames: list, records: dict, docs: dict, docs_index, match: dict, death: dict | None) -> dict:
    match_keys = {docs_index.key(p['champion']) for p in riot_facts.participants(match).values()} - {None}
    killer = docs_index.key(death.get('killer')) if death else None
    return {'frames': {f['id'] for f in frames}, 'records': records or {}, 'docs': docs or {},
            'patch': _major_minor(riot_facts.game_version(match)), 'names': docs_index.names(),
            'match_champions': match_keys, 'killer': killer}


def _major_minor(version) -> str:
    return '.'.join(str(version or '').split('.')[:2])


def _numbers(text: str) -> set:
    return set(re.findall(r'\d+(?:\.\d+)?', text))


def _champions_in(text: str, names: dict) -> set:
    return {key for name, key in names.items() if len(name) >= 2 and name in text}


def check_claim(claim: dict, ctx: dict) -> dict:
    refs = claim.get('refs') or []
    known = [r for r in refs if r in ctx['frames'] or r in ctx['records'] or r in ctx['docs']]
    text = claim.get('text', '')
    if not refs:
        return {'status': 'unsupported', 'reason': '근거 번호 없음'}
    if not known:
        return {'status': 'unsupported', 'reason': f'없는 근거 번호: {", ".join(refs)}'}
    need = KIND_PREFIX.get(claim.get('kind'))
    if need and not any(r.startswith(need) for r in known):
        return {'status': 'unsupported', 'reason': f"{claim.get('kind')} 주장인데 {need} 근거가 없음"}

    outside = _champions_in(text, ctx['names']) - ctx['match_champions']
    if outside:
        return {'status': 'conflict', 'reason': f"경기에 없는 챔피언: {', '.join(sorted(outside))}"}
    for m in KILL_RE.finditer(text):
        word = m.group(1)
        named = {ctx['names'][word]} if word in ctx['names'] else _champions_in(word, ctx['names'])
        if named and ctx['killer'] and ctx['killer'] not in named:
            return {'status': 'conflict', 'reason': f"기록상 처치자는 {ctx['killer']}"}
    cited_docs = [ctx['docs'][r] for r in known if r in ctx['docs']]
    if claim.get('kind') == 'patch' and cited_docs:
        missing = _numbers(text) - set().union(*(_numbers(d['text']) for d in cited_docs))
        if missing:
            return {'status': 'conflict', 'reason': f"문서에 없는 수치: {', '.join(sorted(missing))}"}
    if any(_major_minor(d['patch']) != ctx['patch'] for d in cited_docs):
        return {'status': 'stale', 'reason': f"경기 패치 {ctx['patch']}와 다른 패치 문서"}
    return {'status': 'supported', 'reason': ''}


def verify(coaching: dict, ctx: dict) -> dict:
    claims = [{**c, **check_claim(c, ctx)} for c in coaching.get('claims', [])]
    counts = {s: sum(c['status'] == s for c in claims) for s in ('supported', 'unsupported', 'conflict', 'stale')}
    return {'claims': claims, 'counts': counts,
            'support_rate': round(counts['supported'] / len(claims), 3) if claims else None}


def apply(coaching: dict, checked: dict) -> dict:
    """Keep supported claims; the rest go to held_claims with the reason."""
    return {**coaching, 'claims': [c for c in checked['claims'] if c['status'] == 'supported'],
            'held_claims': [c for c in checked['claims'] if c['status'] != 'supported']}
