"""Test Qwen visual understanding using a single LoL frame (no OCR/YOLO)."""
import base64
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

IMAGE_PATH = ROOT / 'Qwen_Test' / 'frames' / 'frame_000.jpg'
OUTPUT_DIR = ROOT / 'results' / 'qwen_test'

PROMPT = """이 이미지는 리그 오브 레전드(LoL) 게임 영상에서 추출한 프레임입니다.
오직 이미지에서 실제로 확인되는 내용에 근거해 한국어로 답하세요.

1. 화면에서 확인되는 챔피언, 아군/적군, 위치와 현재 상황을 설명하세요.
2. HUD에서 직접 읽을 수 있는 게임 시간, 체력, 스킬 상태, KDA, 미니맵 정보를 알려주세요.
3. 각 판단을 '확실', '불확실', '판독 불가' 중 하나로 구분하세요.
4. 정지 이미지 한 장으로 확인할 수 없는 행동의 원인, 향후 결과, 플레이 실수는 추측하지 마세요.
5. 마지막에 이 프레임만으로 코칭할 수 있는 내용과 추가 프레임이 필요한 내용을 구분하세요.
"""


def main():
    api_key = os.getenv('DASHSCOPE_API_KEY', '').strip()
    base_url = os.getenv('DASHSCOPE_BASE_URL', '').strip()
    model = os.getenv('QWEN_MODEL', '').strip()

    missing = [name for name, value in (
        ('DASHSCOPE_API_KEY', api_key),
        ('DASHSCOPE_BASE_URL', base_url),
        ('QWEN_MODEL', model),
    ) if not value]
    if missing:
        raise SystemExit('Missing .env settings: ' + ', '.join(missing))

    if not IMAGE_PATH.is_file():
        raise SystemExit(f'Image not found: {IMAGE_PATH}')

    image_base64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode('ascii')
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    print(f'Image: {IMAGE_PATH.name}')
    print(f'Model: {model}')
    print('Sending one image to Qwen...')

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': PROMPT},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{image_base64}'
                            },
                        },
                    ],
                }
            ],
            max_tokens=1200,
        )
    except Exception as exc:
        print(f'API call failed ({type(exc).__name__}): {exc}')
        raise SystemExit(1) from exc

    answer = response.choices[0].message.content or '(No text content returned)'
    print('\n===== QWEN RESULT =====\n')
    print(answer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / 'frame_000_result.txt'
    with path.open('w', encoding='utf-8') as file:
        file.write(f'Model: {model}\nImage: {IMAGE_PATH.name}\n')
        file.write(f'Tested at: {datetime.now().astimezone().isoformat()}\n\n')
        file.write(PROMPT + '\n\n===== RESULT =====\n\n' + answer)
    print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()
