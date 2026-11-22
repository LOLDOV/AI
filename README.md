유저 정보, 경기 데이터, 용량이 큰 파일 등을 포함하고 있기 때문에 Data, Videos, result 폴더 Git 하지 않음 (구글 드라이브로 공유할 것.)

## AI Pipeline — VOD Coaching Scene Discovery


### Overview

본 프로젝트는 League of Legends VOD를 분석하여 플레이어의 실력 향상에 도움이 되는 주요 장면을 탐지하고, VLM과 게임 지식 기반 코칭 리포트를 생성하는 AI 파이프라인입니다.

목표: 단순 킬·데스 탐지를 넘어 교전, 스킬 회피, 피격, 포지셔닝, 생존 및 운영 판단 등 코칭 가치가 있는 장면을 분석


### Architecture

LoL VOD (.mp4)
    v
Video Preprocessing (OpenCV)
    v
Full-Video Scene Scouting (Gemini VLM)
    v
Candidate Scoring & Ranking
    v
Dense Frame Verification (Gemini VLM)
    v
Top-K Coaching Scene Selection
    v
Scene Preparation & HUD Analysis
    v
VLM Coaching (Baseline / Assisted)
    v
Coaching Report & Human Evaluation


### Implementation

| Module | Description |
| `pipeline/cli.py` | 통합 실행 명령어 |
| `pipeline/core.py` | 영상 및 프레임 전처리 |
| `pipeline/combat_discovery.py` | 코칭 장면 탐지 및 중요도 평가 |
| `pipeline/analyze.py` | HUD 분석 실행 |
| `pipeline/evidence.py` | 스킬·CC 등 보조 근거 통합 |
| `pipeline/experiment.py` | Gemini 기반 코칭 A/B 분석 |
| `pipeline/scoring.py` | 코칭 결과에 대한 사람 평가 집계 |
| `pipeline/spells.py` | 소환사 주문 식별 후보 분석 |

RAG 관련 지식은 게임 규칙을 보조 근거로 제공하며, 영상에서 직접 관찰한 사실과 구분함.


### Scene Discovery

탐지 파이프라인은 영상 전체를 일정 길이의 윈도우로 분할한 후, VLM을 통해 장면 유형과 중요도를 평가합니다.

(진행중) **v4 개선 사항**

- 교전 외 코칭 관련 장면 탐지 지원
- 스킬 회피 및 포지셔닝 관련 후보 점수화
- 비교전 후보의 2차 검증 진입 허용
- 전 구간 후보 평가 및 Top-K 선정
- API 응답 캐싱 및 중단된 탐색 재개 지원


### Execution

터미널에서 실행하시면 됩니다.

1. API 호출 없는 사전 실행

```powershell
python -m pipeline.cli discover --video videos/LOL_002.mp4 --discovery-id LOL_002_COACH_V4 --champion Kaisa --role adc
```

2. Gemini API 기반 실제 분석

```powershell
python -m pipeline.cli discover --video videos/LOL_002.mp4 --discovery-id LOL_002_COACH_V4 --champion Kaisa --role adc --execute
```

실제 실행에는 `.env` 파일의 `GEMINI_API_KEY`가 필요하며, API 요금 또는 사용량 제한이 적용될 수 있습니다.


### Output

분석 결과는 `results/discovery/<discovery-id>/`에 저장됩니다.

| Output | Description |
| `scan_status.json` | 분석 진행 상태 및 오류 |
| `all_windows.csv` | 전체 윈도우의 1차 분류 결과 |
| `verified_windows.csv` | 2차 검증 결과 |
| `all_candidates.csv` | 검증된 후보 및 점수 |
| `selected_scenes.csv` | 최종 선정 장면 |
| `discovery.json` | 구조화된 최종 탐지 결과 |
| `selected_contact_sheet.jpg` | 선정 장면 시각화 |

일부 파일은 분석 단계가 완료된 후에만 생성됩니다.


### Evaluation

`LOL_002`에서 수작업으로 선정한 3개의 코칭 장면을 기준으로 탐지 결과를 비교하고 있습니다.

| Ground Truth | Importance | Description |
| 00:50–01:35 | 3 | 부쉬 포지셔닝 및 자야 처치 |
| 05:12–05:30 | 1 | 스킬 회피 및 아군 사망 |
| 11:30–11:43 | 2 | 바텀 교전 및 갱킹 대응 |

현재 v4는 전체 39개 윈도우 중 24개의 1차 분석을 완료했습니다.

`429 RESOURCE_EXHAUSTED` 및 타임아웃으로 일부 구간이 미분석 상태이며, 최종 성능 평가는 완료되지 않았습니다.


### Next Steps

1. v4 전체 VOD 탐색 및 2차 검증 완료
2. v3와 v4의 장면 탐지 성능 비교
3. 다른 VOD를 이용한 재현율 및 일반화 검증
4. HUD 보조 정보의 코칭 품질 기여도 평가
5. RAG 기반 코칭 리포트 통합 및 검증

> **Status:** Core pipeline implemented; v4 validation in progress. Candidate scores are not validated accuracy metrics.


### Notes

- 수작업 정답 데이터는 모델 입력이 아닌 독립 평가용으로 사용합니다.
- `results/`, `videos/`, `.env` 등은 GitHub에 업로드하지 않습니다.
- Gemini API 무료 사용량 제한으로 인해 실행이 중단될 수 있습니다.
- 중단 후 동일한 discovery ID로 재실행하면 저장된 성공 배치를 재사용할 수 있습니다.
