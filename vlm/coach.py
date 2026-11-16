import json
from pathlib import Path

# 코칭 리포트 생성.
#  - 장면별 코칭: VLM 분석이 있으면 그 결과(실수·대안)를, 없으면 킬 수지 기반 규칙 코칭을 쓴다.
#  - 경기 총평: LLM을 쓸 수 있으면 장면 분석을 모아 요약하고, 아니면 구간별 킬 격차로 규칙 기반 요약.
#  - 질의응답: 리포트를 근거로 LLM이 답한다 (API 키 필요).
# 규칙 기반 코칭은 영상에서 확인한 사실(킬 시각·팀)만 쓰고, 원인 판단은 '확인할 점'으로 남긴다.

PHASES = [('초반', 0, 600), ('중반', 600, 1200), ('후반', 1200, 10 ** 6)]
TEAM_KO = {'blue': '블루', 'red': '레드'}

GAME_PROMPT = """당신은 리그 오브 레전드 코치입니다. 아래는 한 경기 리플레이에서 자동 검출한 사실 데이터와 장면별 분석입니다.
이 데이터에 근거해서만 경기 총평을 작성하세요. 데이터에 없는 내용(챔피언 조합, 아이템 등)은 지어내지 마세요.
장면을 언급할 때는 장면 번호(#순위)와 게임 시간을 함께 쓰세요.

출력은 아래 JSON 하나만, 코드블록 없이 작성하세요.
{
  "overall": "경기 흐름 총평 3~4문장",
  "strengths": ["잘한 점 (근거 장면 포함)"],
  "improvements": [{"title": "개선점 한 줄", "detail": "구체적 설명과 다음 경기에서 할 행동", "scenes": [순위 숫자]}],
  "practice": ["다음 경기 전에 연습할 것"]
}
"""

QA_PROMPT = """당신은 리그 오브 레전드 코치입니다. 아래 경기 분석 리포트(JSON)만 근거로 사용자의 질문에 한국어로 답하세요.
리포트로 확인할 수 없는 내용은 '이 리포트로는 확인할 수 없다'고 말하고, 추측이면 추측이라고 밝히세요.
관련 장면이 있으면 장면 번호(#순위)와 게임 시간을 함께 알려주세요."""


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def mmss(sec):
    sec = max(0, int(round(sec)))
    return f'{sec // 60:02d}:{sec % 60:02d}'


def side_labels(team):
    """(ally, enemy) team keys; when the player's team is unknown, blue is shown as 'ally' side."""
    if team == 'red':
        return 'red', 'blue'
    return 'blue', 'red'


def scene_kills(scene, ally, enemy):
    a = sum(e.get('kills', 0) for e in scene['events'] if e['source'] == 'video_score' and e['team'] == ally)
    e = sum(e.get('kills', 0) for e in scene['events'] if e['source'] == 'video_score' and e['team'] == enemy)
    first_enemy = min((ev['game_sec'] for ev in scene['events']
                       if ev['source'] == 'video_score' and ev['team'] == enemy and ev.get('game_sec') is not None),
                      default=None)
    return a, e, first_enemy


def rule_scene_coaching(scene, ally, enemy, known_team):
    a, e, first_enemy = scene_kills(scene, ally, enemy)
    us = '아군' if known_team else TEAM_KO[ally]
    them = '상대' if known_team else TEAM_KO[enemy]
    when = scene['game_time'].replace('~', '–')

    if scene['event_free']:
        return {
            'verdict': 'skirmish', 'title': '킬 없는 교전',
            'detail': f'{when} 킬은 나오지 않았지만 교전 신호가 강했던 구간입니다. '
                      '체력·소환사 주문을 쓰고 얻은 것이 있었는지(라인·오브젝트·상대 귀환), 아니면 손해만 봤는지 확인하세요.',
            'checks': ['교전 시작 전 체력·마나와 주요 스킬 상태', '교전 후 누가 먼저 귀환했는지', '교전으로 얻은 라인·오브젝트 이득'],
            'ally_kills': 0, 'enemy_kills': 0, 'severity': 0,
        }
    if e > a:
        first = f' 첫 실점은 {mmss(first_enemy)}입니다.' if first_enemy is not None else ''
        return {
            'verdict': 'loss', 'title': f'교전 패배 · {us} {a} : {e} {them}',
            'detail': f'{when} 구간에서 {e}킬을 내주고 {a}킬을 얻었습니다.{first} '
                      '첫 실점 직전 프레임에서 위치·시야·상대 합류 여부를 확인하세요.',
            'checks': ['첫 실점 전 미니맵에 상대 정글·서포터가 보였는가', '교전에 들어갈 때 아군 합류가 가능한 거리였는가',
                       '후퇴 경로에 시야(와드)가 있었는가'],
            'ally_kills': a, 'enemy_kills': e, 'severity': e - a,
        }
    if a > e:
        return {
            'verdict': 'win', 'title': f'교전 승리 · {us} {a} : {e} {them}',
            'detail': f'{when} 구간에서 {a}킬을 얻었습니다. 이득 이후 20~30초 안에 드래곤·전령·타워로 전환했는지 확인하세요.',
            'checks': ['교전 직후 가까운 오브젝트 상태', '귀환 타이밍과 라인 상태', '다음 교전까지의 시야 확보'],
            'ally_kills': a, 'enemy_kills': e, 'severity': e - a,
        }
    return {
        'verdict': 'trade', 'title': f'킬 교환 · {us} {a} : {e} {them}',
        'detail': f'{when} 구간에서 킬을 {a}:{e}로 주고받았습니다. 교환 이후 어느 팀이 더 많은 오브젝트·라인 이득을 챙겼는지 확인하세요.',
        'checks': ['교환에 참여한 아군 수와 상대 수', '교환 직후의 오브젝트·라인 상태'],
        'ally_kills': a, 'enemy_kills': e, 'severity': 0,
    }


def vlm_scene_coaching(base, analysis):
    if not analysis:
        return base
    result = dict(base)
    result['source'] = 'vlm'
    result['situation'] = analysis.get('situation')
    result['mistakes'] = analysis.get('mistakes') or []
    result['alternatives'] = analysis.get('alternatives') or []
    result['key_moment'] = analysis.get('key_moment')
    result['player_champion'] = analysis.get('player_champion')
    return result


def phase_stats(kills, ally, enemy):
    stats = []
    for name, start, end in PHASES:
        a = sum(k['kills'] for k in kills if k['team'] == ally and start <= (k.get('game_sec') or 0) < end)
        e = sum(k['kills'] for k in kills if k['team'] == enemy and start <= (k.get('game_sec') or 0) < end)
        if a or e:
            stats.append({'phase': name, 'range': f'{start // 60}~{end // 60 if end < 10 ** 5 else ""}분',
                          'ally': a, 'enemy': e, 'diff': a - e})
    return stats


def rule_game_summary(ctx):
    us = '아군' if ctx['known_team'] else TEAM_KO[ctx['ally']]
    a_total, e_total = ctx['final']
    phases = ctx['phases']
    scenes = ctx['scene_coaching']
    losses = sorted((s for s in scenes if s['verdict'] == 'loss'), key=lambda s: -s['severity'])
    wins = sorted((s for s in scenes if s['verdict'] == 'win'), key=lambda s: s['severity'])
    skirmishes = [s for s in scenes if s['verdict'] == 'skirmish']

    flow = ', '.join(f"{p['phase']} {p['ally']}:{p['enemy']}" for p in phases)
    worst_phase = min(phases, key=lambda p: p['diff']) if phases else None
    best_phase = max(phases, key=lambda p: p['diff']) if phases else None
    result_text = '앞선' if a_total > e_total else '뒤처진' if a_total < e_total else '비슷한'
    overall = (f'{us} 기준 최종 킬 {a_total}:{e_total}로 킬 수에서 {result_text} 경기입니다. 구간별 킬은 {flow}입니다. '
               f'교전 패배 장면 {len(losses)}개, 승리 장면 {len(wins)}개, 킬 없는 교전 {len(skirmishes)}개를 찾았습니다.')

    improvements = []
    if worst_phase and worst_phase['diff'] < 0:
        in_phase = [s['rank'] for s in losses if s['phase'] == worst_phase['phase']][:3]
        improvements.append({
            'title': f"{worst_phase['phase']}({worst_phase['range']}) 킬 격차 {worst_phase['diff']}",
            'detail': f"{worst_phase['phase']}에 {worst_phase['enemy']}킬을 내주고 {worst_phase['ally']}킬을 얻었습니다. "
                      '이 구간의 실점 장면을 다시 보며 교전을 열기 전 위치와 상대 동선을 확인하세요.',
            'scenes': in_phase,
        })
    if losses:
        worst = losses[0]
        improvements.append({
            'title': f"가장 큰 실점 장면 #{worst['rank']} ({worst['game_time']})",
            'detail': worst['detail'],
            'scenes': [worst['rank']],
        })
    if len(losses) >= 3:
        improvements.append({
            'title': f'교전 패배가 {len(losses)}번 반복됨',
            'detail': '패배한 교전들의 공통점(같은 라인, 같은 시간대, 시야 없는 지역 진입)을 찾아보세요. '
                      '반복되는 패턴이 가장 먼저 고칠 습관입니다.',
            'scenes': [s['rank'] for s in losses[:4]],
        })
    if len(skirmishes) >= 2:
        improvements.append({
            'title': f'킬 없는 교전 {len(skirmishes)}회',
            'detail': '킬로 이어지지 않은 소모전이 여러 번 있었습니다. 교전 후 체력 손해로 귀환하면서 라인·오브젝트를 잃지 않았는지 확인하세요.',
            'scenes': [s['rank'] for s in skirmishes],
        })

    strengths = []
    if best_phase and best_phase['diff'] > 0:
        strengths.append(f"{best_phase['phase']}({best_phase['range']})에 킬 {best_phase['ally']}:{best_phase['enemy']}로 앞섰습니다.")
    for s in wins[-2:][::-1]:
        strengths.append(f"#{s['rank']} ({s['game_time']}) 교전에서 {s['ally_kills']}킬을 얻었습니다.")

    practice = []
    if losses:
        practice.append(f"실점 장면 #{losses[0]['rank']}의 첫 실점 10초 전부터 0.5배속으로 다시 보며 미니맵만 따라가기")
    practice.append('교전 전 3초 체크: 미니맵 상대 위치 · 내 주요 스킬 쿨다운 · 아군 합류 거리')
    if wins:
        practice.append('킬을 얻은 뒤 가까운 오브젝트를 먼저 확인하는 습관 만들기')

    return {'overall': overall, 'strengths': strengths, 'improvements': improvements, 'practice': practice}


def llm_game_summary(ctx, vlm_items):
    from vlm.scene_analyzer import client_from_env, parse_json
    client, model = client_from_env()
    payload = {
        'player_team': ctx['ally'] if ctx['known_team'] else '판독 불가',
        'final_kills_ally_enemy': ctx['final'],
        'phases': ctx['phases'],
        'scenes': [{k: s.get(k) for k in ('rank', 'game_time', 'title', 'detail', 'situation', 'mistakes', 'alternatives')}
                   for s in ctx['scene_coaching']],
    }
    response = client.chat.completions.create(
        model=model, max_tokens=1500,
        messages=[{'role': 'user', 'content': GAME_PROMPT + '\n\n' + json.dumps(payload, ensure_ascii=False)}],
    )
    return parse_json(response.choices[0].message.content or '')


def build_coaching(result_dir, use_llm=False):
    result_dir = Path(result_dir)
    scenes = read_json(result_dir / 'scenes.json', {'all_scenes': []})
    kills = read_json(result_dir / 'kills.json', [])
    player = read_json(result_dir / 'player.json', {}) or {}
    vlm = {v['rank']: v.get('analysis') for v in read_json(result_dir / 'vlm.json', []) or []}

    known_team = bool(player.get('team'))
    ally, enemy = side_labels(player.get('team'))
    phases = phase_stats(kills, ally, enemy)
    final = (
        max((k['score_after'] for k in kills if k['team'] == ally), default=0),
        max((k['score_after'] for k in kills if k['team'] == enemy), default=0),
    )

    scene_coaching = []
    for scene in sorted(scenes['all_scenes'], key=lambda s: s['rank']):
        if not scene.get('frames'):
            continue
        base = rule_scene_coaching(scene, ally, enemy, known_team)
        base.update({'rank': scene['rank'], 'game_time': scene['game_time'], 'source': 'rule'})
        start_game = next((e['game_sec'] for e in scene['events'] if e.get('game_sec') is not None), None)
        base['phase'] = next((name for name, s, e in PHASES if start_game is not None and s <= start_game < e), None)
        scene_coaching.append(vlm_scene_coaching(base, vlm.get(scene['rank'])))

    ctx = {'ally': ally, 'enemy': enemy, 'known_team': known_team, 'final': final,
           'phases': phases, 'scene_coaching': scene_coaching}
    summary, source, error = None, 'rule', None
    if use_llm:
        try:
            summary = llm_game_summary(ctx, vlm)
            source = 'llm' if summary else 'rule'
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    if not summary:
        summary = rule_game_summary(ctx)

    return {
        'source': source,
        'error': error,
        'player': player,
        'ally': ally,
        'known_team': known_team,
        'final': final,
        'phases': phases,
        **summary,
        'scenes': scene_coaching,
    }


def answer_question(result_dir, question, history=None):
    from vlm.scene_analyzer import client_from_env
    client, model = client_from_env()
    coaching = read_json(Path(result_dir) / 'coaching.json', {})
    context = {k: coaching.get(k) for k in ('player', 'final', 'phases', 'overall', 'improvements', 'scenes')}
    messages = [{'role': 'system', 'content': QA_PROMPT + '\n\n리포트:\n' + json.dumps(context, ensure_ascii=False)}]
    for turn in (history or [])[-6:]:
        if turn.get('role') in ('user', 'assistant') and turn.get('content'):
            messages.append({'role': turn['role'], 'content': str(turn['content'])[:2000]})
    messages.append({'role': 'user', 'content': question[:2000]})
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=900)
    return response.choices[0].message.content or ''
