"""Death-scene coaching with numbered evidence: F (frames), R (Riot records), D (patch docs).

Frame capture and the Gemini request reuse RAG/coaching.py. Leaving out records or docs
gives the ablation conditions with the same frames.
"""
from __future__ import annotations

from pathlib import Path

from . import riot_facts
from .timesync import mmss

PROMPT = '''당신은 리그 오브 레전드 코치입니다. 대상은 골드 이하 플레이어이고, 아래는 플레이어({champion})가 {what} 장면입니다.
규칙
- 주장(claim)마다 근거 ID를 refs에 붙이세요: 프레임 F*{ref_rule}
- kind: frame=화면에서 직접 본 것, record=경기 기록에 있는 사실, patch=패치 문서의 내용, advice=조언(근거가 된 ID를 refs에).
- 화면에 보이지 않는 스킬 시전, 쿨타임, 체력 숫자는 단정하지 마세요. 패치 수치는 D 문서에 있는 숫자만 쓰세요.
- 경기 기록에는 스킬 시전·회피·시야 정보가 없습니다. 패치 문서는 스킬 설명일 뿐 영상에서 썼다는 증거가 아닙니다.
- 원인(cause)은 한 문장, 대안 행동은 다음 경기에서 바로 해볼 수 있게 구체적으로 쓰세요.
- verdict: improve(개선 필요) | check(판단 근거 부족) | best(잘 대처함)
JSON만 반환:
{{"title":"짧은 제목","situation":"상황 요약","cause":"사망 원인","verdict":"improve",
 "claims":[{{"id":"c1","kind":"frame","text":"...","refs":["F3"]}}],
 "alternatives":[{{"action":"...","reason":"...","refs":["F2"]}}],
 "checks":["영상을 다시 볼 때 확인할 점"]}}

[프레임] (영상 시각 / 기준 시점 대비 초)
{frames}
{records}{docs}'''


def record_lines(death: dict, match: dict, timeline: dict, player: dict, local_name=lambda c: c) -> dict:
    """{'R1': text, ...} describing the death and the minute before it."""
    people = riot_facts.participants(match)
    champs = {p['champion'] for p in people.values()}
    n = lambda c: f'{local_name(c)}({c})' if c else '미니언·포탑·몬스터'
    allies = [n(p['champion']) for p in people.values()
              if p['team'] == player['team'] and p['participant_id'] != player['participant_id']]
    enemies = [n(p['champion']) for p in people.values() if p['team'] != player['team']]
    lines = [f"플레이어 {n(player['champion'])} {player['position']} / {player['team']} 팀. "
             f"아군: {', '.join(allies)}. 적: {', '.join(enemies)}",
             f"사망 시각 게임 {mmss(death['game_sec'])}. 처치: {n(death['killer'])}. "
             f"어시스트: {', '.join(n(a) for a in death['assists']) or '없음'}."]
    if death['damage_received']:
        lines.append('사망 직전 받은 피해(큰 순): ' + ', '.join(
            f"{n(d['source']) if d['source'] in champs else d['source']} {d['slot']} {d['damage']}"
            for d in death['damage_received']))
    st = death.get('state')
    if st:
        lines.append(f"게임 {mmss(st['snapshot_game_sec'])} 기준(1분 단위 기록) 레벨 {st['level']}, 보유 골드 {st['current_gold']}.")
    for e in riot_facts.events_between(match, timeline, death['game_sec'] - 60, death['game_sec'] + 10):
        if e['type'] == 'CHAMPION_KILL':
            if abs(e['game_sec'] - death['game_sec']) < 0.5 and e.get('victim') == player['champion']:
                continue
            lines.append(f"게임 {mmss(e['game_sec'])}: {n(e['killer'])} → {n(e['victim'])} 처치")
        elif e['type'] == 'ELITE_MONSTER_KILL':
            lines.append(f"게임 {mmss(e['game_sec'])}: {e['team']} 팀 {e['monster']} 처치")
        else:
            lines.append(f"게임 {mmss(e['game_sec'])}: {e['type']} {e.get('building') or ''} {e.get('lane') or ''}".strip())
    return {f'R{i}': line for i, line in enumerate(lines, 1)}


def doc_lines(docs: list) -> dict:
    """{'D1': {'patch', 'text', 'id', 'source_url'}, ...} from PatchDocs.for_death()."""
    return {f'D{i}': {'patch': d['patch_version'], 'text': d['text'], 'id': d['id'], 'source_url': d['source_url']}
            for i, d in enumerate(docs, 1)}


def build_prompt(champion: str, scene: dict, frames: list, records: dict, docs: dict) -> str:
    what = '사망한' if scene.get('type') == 'death' else '위험에 빠졌거나 중요한 판단을 한'
    rule = (', 경기 기록 R*' if records else '') + (', 패치 문서 D*' if docs else '')
    fl = '\n'.join(f"{f['id']}: 영상 {mmss(f['video_sec'])} / {f['rel']:+.0f}초" for f in frames)
    rl = ('\n[경기 기록 - Riot API]\n' + '\n'.join(f'{k}: {v}' for k, v in records.items()) + '\n') if records else ''
    dl = ('\n[패치 문서 - Data Dragon]\n' + '\n'.join(
        f"{k}: [{d['patch']}] {d['text'][:420]}" for k, d in docs.items()) + '\n') if docs else ''
    return PROMPT.format(champion=champion, what=what, ref_rule=rule, frames=fl, records=rl, docs=dl)


def normalize(obj) -> dict:
    obj = obj if isinstance(obj, dict) else {}
    refs = lambda x: [str(r) for r in (x.get('refs') or [])][:6]
    claims = [{'id': str(c.get('id') or f'c{i}'), 'kind': c.get('kind'), 'text': str(c.get('text', ''))[:400], 'refs': refs(c)}
              for i, c in enumerate(obj.get('claims') or [], 1) if isinstance(c, dict) and c.get('text')]
    alts = [{'action': str(a.get('action', ''))[:200], 'reason': str(a.get('reason', ''))[:300], 'refs': refs(a)}
            for a in (obj.get('alternatives') or [])[:3] if isinstance(a, dict) and a.get('action')]
    verdict = obj.get('verdict') if obj.get('verdict') in ('improve', 'check', 'best') else 'check'
    return {'title': str(obj.get('title', ''))[:60], 'situation': str(obj.get('situation', ''))[:500],
            'cause': str(obj.get('cause', ''))[:300], 'verdict': verdict, 'claims': claims,
            'alternatives': alts, 'checks': [str(x)[:200] for x in (obj.get('checks') or [])][:3]}


def capture(video: Path, scene: dict, dest: Path, count: int = 8) -> list:
    """Evenly spaced frames over the scene, labelled F1..Fn."""
    from RAG.coaching import _capture_frames
    rows = _capture_frames(video, {'scene_id': scene['scene_id'], 'start_mp4_sec': scene['start_sec'],
                                   'end_mp4_sec': scene['end_sec']}, dest, count)
    frames = []
    for i, (sid, fr, path) in enumerate(rows):
        t = scene['start_sec'] + (scene['end_sec'] - scene['start_sec']) * i / max(1, count - 1)
        fid = f'F{i + 1}'
        frames.append({'id': fid, 'video_sec': round(t, 2), 'rel': t - scene['anchor_sec'], 'file': str(path),
                       'request': (sid, {**fr, 'frame_id': fid}, path)})
    return frames


def generate(model: str, champion: str, scene: dict, frames: list, records: dict, docs: dict):
    """Returns (coaching, reply). reply has latency_sec and token_usage."""
    from RAG.coaching import _coaching_request
    prompt = build_prompt(champion, scene, frames, records, docs)
    reply = _coaching_request(model, prompt, [f['request'] for f in frames])
    return normalize(reply['parsed']), reply
