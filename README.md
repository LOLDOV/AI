유저 정보 및 경기 데이터 json를 포함하고 있기 때문에 Data 폴더 Git 하지 않음

구글 드라이브로 공유할 것.

같은 이유로 qwen 폴더 내 이벤트 json과 frame 사진, mp4 등도 구글 드라이브로 공유 할 것.

`results/` 폴더(프레임 이미지에 소환사명이 보임)도 Git 하지 않음.

## VOD 분석 파이프라인

중간발표 처리 흐름(장면 후보 선정 → 영상 분석 → 코칭) 중 VOD 쪽 단계를 구현한 것.

```
스캔(HUD OCR·시각 신호) → 시간 동기화 → 킬 검출 → 장면 후보 → 대표 프레임 → (선택) Qwen VLM 분석
```

| 모듈 | 역할 |
| --- | --- |
| `vod/video.py` | ffprobe 메타데이터, ffmpeg 파이프로 1fps 프레임 샘플링 |
| `vod/hud.py` | 상단 HUD 게임 시계·킬 스코어 OCR (RapidOCR = PaddleOCR 모델의 ONNX 버전) |
| `vod/scan.py` | 초 단위 신호: 게임 시계, 킬 스코어, 화면 변화량, 체력바 색 밀집도 |
| `vod/sync.py` | 게임 시간 ↔ 영상 시간 offset 구간 계산 (일시정지·배속 시 구간 분리) |
| `vod/events.py` | 킬 스코어 증가로 킬 이벤트 검출 (Riot 타임라인 없이 동작) |
| `vod/scenes.py` | 이벤트 기반 + 이벤트 없는 교전(시각 신호) 장면 후보, 병합·순위·대표 프레임 |
| `vod/frames.py` | 장면별 대표 프레임 JPG 저장 |
| `vlm/scene_analyzer.py` | 시간순 프레임 + 검출 이벤트를 Qwen에 넣어 JSON 분석 (주장별 근거 프레임·확신도) |
| `evaluation/scene_eval.py` | 장면 탐지 Precision/Recall (시간 IoU ≥ 0.5, 1:1 매칭, 유형별 Recall) |
| `evaluation/sync_eval.py` | Riot 킬 시각 vs 영상 검출 킬 시각 오차(중앙값·최댓값) |
| `vod/inventory.py` | 수집 VOD 수량·해상도·길이 집계 → `results/vod_inventory.csv` |

### 설치

ffmpeg 필요 (`brew install ffmpeg` / Windows는 ffmpeg 설치 후 PATH 등록). Python 3.12 권장.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 실행

드라이브의 `롤 리플 영상` 폴더 mp4를 `data/vod/`에 둔다.

```
python -m vod.inventory data/vod
python main.py data/vod/LOL_002.mp4                 # 장면 후보 + 프레임
python main.py data/vod/LOL_002.mp4 --dry-run       # VLM 프롬프트만 생성
python main.py data/vod/LOL_002.mp4 --vlm           # Qwen 호출 (.env 필요)
python main.py data/vod/LOL_002.mp4 --timeline data/riot/timelines/<matchId>.json   # Riot 이벤트 병합 + 동기화 검증
python -m evaluation.scene_eval labels/scenes.csv --top-k 10
```

결과: `results/<영상이름>/` 아래 `scan.json`(초 단위 신호, 캐시), `sync.json`, `kills.json`, `scenes.json`, `frames/`, `vlm.json`, `run_meta.json`(설정·소요 시간).

`labels/scenes_template.csv`는 정답 CSV 형식 예시. 실제 평가용 정답 구간은 사람이 직접 라벨링해야 한다.

### 참고·한계

- HUD 좌표는 1282x832 리플레이 녹화본 기준이며 시계 글자 높이로 스케일한다. 다른 해상도/HUD 배율은 `vod/hud.py`의 상대 좌표 확인 필요.
- 킬 스코어는 팀 전체 킬이라 분석 대상 플레이어가 관여하지 않은 킬도 장면 후보가 된다. 킬 알림·킬 피드 OCR 또는 Riot 타임라인 participantId로 필터링하는 것이 다음 단계.
- 시각 신호(체력바 밀집·화면 변화) 가중치와 임계값은 실험값. 개발 영상으로 조정 후 테스트 전에 고정한다.
