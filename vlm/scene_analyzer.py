import base64
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 주요 장면의 시간순 프레임 + 검출 이벤트를 Qwen(VLM)에 함께 넣어 장면 맥락을 분석한다.
# 응답은 JSON으로 받고, 주장마다 근거 프레임 ID와 확신도를 붙이게 해 이후 코칭 검증 단계에서 대조한다.

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

# Gemini 등 '생각' 토큰도 max_tokens에 포함하는 모델은 .env의 VLM_THINKING_TOKENS 만큼 한도를 늘린다
THINKING_TOKENS = int(os.getenv('VLM_THINKING_TOKENS', '0'))
MAX_TOKENS = 1800 + THINKING_TOKENS

PROMPT = """당신은 리그 오브 레전드(LoL) 리플레이를 복기하는 코치입니다.
아래 이미지는 한 장면의 프레임을 시간순으로 나열한 것이고, 각 이미지 앞에 프레임 ID와 게임 시간이 붙어 있습니다.
함께 제공되는 '검출 이벤트'는 HUD 킬 스코어 OCR·Riot 타임라인에서 얻은 사실 데이터입니다.

규칙
- 이미지와 검출 이벤트로 확인되는 내용만 근거로 삼으세요. 보이지 않는 것은 추측하지 말고 '판독 불가'로 두세요.
- 모든 주장에 근거 프레임 ID(evidence)와 확신도(confidence: 확실/불확실)를 붙이세요.
- 화면 중앙(카메라가 따라가는) 챔피언과 좌하단 HUD 초상화의 챔피언을 '분석 대상 플레이어'로 봅니다.
- 대안 행동은 이 장면에서 실제로 선택 가능했던 구체적 행동으로 제시하세요.

출력은 아래 JSON 하나만, 코드블록 없이 작성하세요.
{
  "player_champion": "분석 대상 챔피언 이름 또는 판독 불가",
  "location": "맵 위치(예: 바텀 강가, 미드 1차 타워 앞)",
  "situation": "장면 전체 요약 2~3문장",
  "timeline": [{"frame": "프레임ID", "game_time": "mm:ss", "observation": "관찰 내용", "confidence": "확실|불확실"}],
  "key_moment": {"frame": "프레임ID", "description": "승패를 가른 순간"},
  "mistakes": [{"description": "실수 또는 아쉬운 판단", "evidence": ["프레임ID"], "confidence": "확실|불확실"}],
  "alternatives": [{"action": "대안 행동", "reason": "근거", "evidence": ["프레임ID"]}],
  "needs_more_info": ["판단에 추가로 필요한 정보 (이전 프레임, 쿨다운, 시야 등)"]
}
"""


def vlm_available():
    return all(os.getenv(k, '').strip() for k in ('DASHSCOPE_API_KEY', 'DASHSCOPE_BASE_URL', 'QWEN_MODEL'))


def client_from_env():
    api_key = os.getenv('DASHSCOPE_API_KEY', '').strip()
    base_url = os.getenv('DASHSCOPE_BASE_URL', '').strip()
    model = os.getenv('QWEN_MODEL', '').strip()
    missing = [name for name, value in (
        ('DASHSCOPE_API_KEY', api_key),
        ('DASHSCOPE_BASE_URL', base_url),
        ('QWEN_MODEL', model),
    ) if not value]
    if missing:
        raise RuntimeError('Missing .env settings: ' + ', '.join(missing))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=180.0), model


def summarize_events(scene):
    lines = []
    for e in scene['events']:
        if e['source'] == 'video_score':
            game = e.get('game_sec')
            when = f"게임 {int(game) // 60:02d}:{int(game) % 60:02d}" if game is not None else f"영상 {e['video_sec']:.0f}s"
            lines.append(f"- {when}: {e['team']} 팀 킬 +{e['kills']} (킬 스코어 {e['score_after']})")
        elif e['source'] == 'riot_timeline':
            ms = e.get('timestamp_ms', 0) // 1000
            lines.append(f"- 게임 {ms // 60:02d}:{ms % 60:02d}: {e.get('type')} "
                         f"killer={e.get('killer_id')} victim={e.get('victim_id')} "
                         f"monster={e.get('monster_type')} building={e.get('building_type')}")
        elif e['source'] == 'visual':
            lines.append(f"- 영상 {e['video_sec']:.0f}s: 이벤트 없음, 교전 신호 강도 {e['intensity']}")
    return '\n'.join(lines) or '- 없음'


def player_line(player):
    if not player or not player.get('team'):
        return '분석 대상 플레이어의 팀: 판독 불가'
    side = '왼쪽' if player['team'] == 'blue' else '오른쪽'
    team = '블루' if player['team'] == 'blue' else '레드'
    return (f"분석 대상 플레이어: {team} 팀 ({side} 초상화 열 {player['slot'] + 1}번째). "
            f"{team} 팀 킬은 아군 킬, 상대 팀 킬은 아군이 잃은 킬입니다.")


def build_messages(scene, frames, player=None):
    header = (
        f"장면 순위 {scene['rank']}, 게임 시간 {scene['game_time']}, "
        f"출처 {', '.join(scene['sources'])}\n{player_line(player)}\n검출 이벤트:\n{summarize_events(scene)}"
    )
    content = [{'type': 'text', 'text': PROMPT + '\n\n' + header}]
    for frame in frames:
        content.append({'type': 'text', 'text': f"[{frame['frame_id']}] 게임 {frame['game_time']}"})
        image_b64 = base64.b64encode(Path(frame['path']).read_bytes()).decode('ascii')
        content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_b64}'}})
    return [{'role': 'user', 'content': content}]


def parse_json(text):
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def analyze_scene(scene, frames, client=None, model=None, dry_run=False, player=None):
    """Returns a dict with the parsed analysis plus call log (model, latency, tokens)."""
    messages = build_messages(scene, frames, player)
    log = {'rank': scene['rank'], 'frames': [f['frame_id'] for f in frames]}

    if dry_run:
        log['dry_run'] = True
        log['prompt_text'] = '\n'.join(c['text'] for c in messages[0]['content'] if c['type'] == 'text')
        return {'analysis': None, 'log': log}

    started = time.monotonic()
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=MAX_TOKENS)
    text = response.choices[0].message.content or ''
    usage = getattr(response, 'usage', None)
    log.update({
        'model': model,
        'latency_sec': round(time.monotonic() - started, 2),
        'prompt_tokens': getattr(usage, 'prompt_tokens', None),
        'completion_tokens': getattr(usage, 'completion_tokens', None),
    })
    analysis = parse_json(text)
    if analysis is None:
        log['parse_error'] = True
        log['raw'] = text
    return {'analysis': analysis, 'log': log}
