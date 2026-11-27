# labels — 정답 라벨과 평가 결과

평가 담당: 김정은 / 기준일: 2026-11-27

## 정답 라벨 (Ground Truth)

| 파일 | 내용 |
|---|---|
| `scenes.csv` | `evaluation/scene_eval.py` 입력 형식. 영상 시간(초) 기준 정답 구간 39개 |
| `gt_scenes.csv` | 장면별 상세 라벨 (데스 33건 + 죽을 뻔함 6건, 중요도 0~5, 어려운 사례 표시) |
| `gt_videos.csv` | 영상 15개 정보 (Match ID, 패치, 게임 00:00의 영상 시간) |
| `gt_hud.csv` | HUD 정답 23장 (게임 시계, 팀 킬, 레벨). `ai_*` 칸은 비어 있음 |
| `gt_time_sync.csv` | 게임 시계 ↔ 영상 시간 기준점 14개 (오프셋 -3 ~ -8초) |

- 장면 종류: 분석 대상 챔피언의 데스 장면과 죽을 뻔한 장면(Riot 기록에 없음 = 이벤트 없는 장면)
- 근거: Match ID가 있는 경기는 Riot Match-V5 기록, 없는 경기는 평가자가 영상의 킬 알림·화면을 직접 확인
- LOL_034, LOL_037은 Match ID를 찾지 못해 라벨링하지 않음

## 웹(배포 서비스) 평가 결과

| 파일 | 내용 |
|---|---|
| `eval_web_detection.csv` | 웹이 고른 장면 60개와 정답 매칭 결과 (시간 = 게임 시계) |
| `eval_web_coaching.csv` | 웹 VLM 코칭 문장 90개 판정 (SUPPORTED / CONTRADICTED / UNSUPPORTED / UNKNOWN / NOT_FACT) |
| `eval_web_summary.csv` | 지표 요약 |

평가 영상: LOL_002, 004, 009, 024, 033 (LOL_018은 웹 예시 목록에서 사라져 제외)

| 지표 | 값 |
|---|---|
| 장면 Recall (정답 데스 시각이 AI 구간 안에 있으면 일치) | 68.8% (11/16) |
| 장면 Precision (같은 기준) | 18.3% (11/60) |
| 장면 Recall (시간 IoU ≥ 0.5) | 0% — AI 구간이 정답보다 길어 IoU 최대 0.47 |
| 근거 지원률 | 79.2% (61/77, 조언 제외) |
| 코칭 오류율 | 18.2% (14/77) |

- 웹은 "킬이 난 모든 교전"을 고르므로 데스 기준 Precision은 낮게 나온다.
- 5가지 비교 조건(full / no_verify / no_rag / no_ocr / vlm_only)은 `link.run` 결과가 필요해 아직 평가하지 않음.
