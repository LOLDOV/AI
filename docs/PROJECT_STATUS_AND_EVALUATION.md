# LOLDOV 개발 현황 및 정답 라벨·평가 프로토콜

> 기준일: 2026-11-24 · 대상: League of Legends 소환사의 협곡 일반/랭크 VOD · 현재 기준 파이프라인: **Gemini v4.0 장면 탐지 + Riot Match/Timeline + 경기 패치 Data Dragon RAG + 로컬 웹 MVP**
>
> **중요:** 아래에서 `구현`, `로컬 모의 테스트 통과`, `사용자 PC 실험 확인`, `미검증`은 서로 다른 상태이다. 모델의 생성 문장과 내부 점수는 사람이 확인한 정답(GT)이 아니다.

## 1. 프로젝트 목표 및 현재 범위

LOLDOV는 League of Legends 경기 영상(MP4)에서 코칭 가치가 있는 장면을 자동으로 선별하고, Gemini VLM의 영상 해석을 Riot Match/Timeline 및 실제 경기 패치의 챔피언·아이템 정보로 보조해 코칭 리포트를 생성하는 프로젝트이다. 초기 목표 사용자는 골드 이하의 플레이어이다. 긍정적 플레이와 개선할 플레이 모두를 다루며, 기록되지 않은 스킬 사용·회피·포지셔닝 등은 영상 근거가 없는 한 단정하지 않는 것이 원칙이다.

현재 개발용 사용자 흐름은 **LOLDOV 자체 계정 회원가입/로그인 → Riot ID 최초 등록(소유권 미검증) → 대전 기록 조회 → 사용자가 해당 경기 선택 → MP4 업로드 → v4 Top-K → Riot/RAG → assisted 코칭 초안**이다. Riot 공식 계정 로그인(RSO/OAuth2), MP4만으로 경기 자동 식별, 상용 서비스 배포는 아직 아니다.

## 2. 시스템 아키텍처

```text
[LOLDOV 로컬 웹 service/]
  자체 회원가입/로그인(≠ Riot RSO)
  Riot ID 1회 등록 → Riot API 경기 카드 표시
  사용자가 '업로드할 영상과 동일한 경기' 선택 → MP4 업로드
                   │
                   ▼
[pipeline/ - v4.0, 기존 소스 유지]
  OpenCV 프레임 추출
  24초 비중첩 Scout 윈도우 / 4초 샘플링
  Gemini Scout 중요도·장면 유형·대상 관련성 추정
  상위 후보 10개(기본) 재검증 / 2초 간격 dense Verify
  랭킹·인접 장면 병합 → Top-K(기본 5개)
                   │
       ┌───────────┴──────────────────┐
       ▼                              ▼
[RIOT/]                          [RAG/]
 Riot ID→PUUID, Match/Timeline     실제 경기 버전→Data Dragon 16.xx.x
 선정 경기·소환사·시간 검증          키아나 QWER/패시브·아이템 정적 설명 검색
 선택적 OCR HUD 시계→오프셋         단순 검색형 RAG(벡터DB 아님)
 킬/어시/사망/오브젝트 기록          공식 패치노트 변경 내역 검색 아님
       └───────────┬──────────────────┘
                   ▼
[RAG/coaching.py + Gemini]
 같은 장면 프레임 + 영상 시각 + [선택적으로 Riot·RAG 근거]
 summary / facts / praise / improvements / unknowns / source_urls
                   │
                   ▼
[service/report 및 사람 평가]
 장면별 코칭 결과·API 지연시간/토큰 → 정답 라벨과 별도 대조
```

## 3. 실제 폴더 및 책임 분리

| 폴더/파일 | 책임 | 현재 상태 |
|---|---|---|
| `pipeline/cli.py`, `pipeline/combat_discovery.py` | VOD 전 구간 Scout→Verify→Top-K, 캐시·재시작 | 구현·LOL_001/002 탐지 실행 확인. **선택 버전 v4.0**, v4.1 실험본은 기본으로 사용하지 않음 |
| `pipeline/core.py`, `pipeline/analyze.py`, `pipeline/engines/` | 장면 프레임·HUD 및 OpenCV 시각 근거 후보 | 실험용. 환경별 보정 필요. 스킬 시전의 확정 증거 아님 |
| `pipeline/experiment.py`, `pipeline/scoring.py` | 기존 영상 단독/기계 보조 A/B, 사람이 매긴 0~2 루브릭 요약 | 코드 존재; 별도 실험. Riot+RAG A/B와 혼동 금지 |
| `RIOT/riot_api.py` | Riot ID 목록 기반 최근 Match/Timeline JSON 저장·캐시 | 기존 구현 |
| `RIOT/client.py` | Riot JSON 로딩, 사용자·킬/어시 등 사건 연결 | 구현; LOL_001 실제 데이터로 연결 확인 |
| `RIOT/match_resolver.py` | 계정 대전 기록 카드, 선택 경기 참가자·영상 길이 검사 | 구현; 모의 테스트 통과. 자동 경기 식별 아님 |
| `RIOT/time_sync.py` | 여러 HUD 시계 OCR 값으로 영상−게임 시간 차이 후보 계산 | 선택적 구현, 3개 이상 같은 ROI 일치 필요; Riot 사건과의 별도 자동 검증은 미구현 |
| `RAG/cli.py` | 선정 장면·Riot 기록·패치 문서 결합, `build/coach` 명령 | 구현 |
| `RAG/patch_rag.py` | 경기 버전 Data Dragon 챔피언·아이템 검색 | 구현; 동적 아이템 수치 오류를 줄이는 정적 정보 필터 적용 |
| `RAG/coaching.py` | Gemini 코칭 Baseline/Assisted, 캐시·지연시간/토큰 저장 | 구현; QWER·킬 시각 과신 억제 프롬프트 보강 |
| `service/app.py`, `service/storage.py`, `service/workflow.py`, `service/static/index.html` | 로컬 자체 로그인, Riot ID 등록, 경기 조회, 영상 업로드, 단일 작업 실행 및 보고서 표시 | 구현·로컬 모의 테스트 8개 통과. 사용자 PC에서 Uvicorn 시작/`GET /` 200 확인. **웹 전체 사용 시나리오 실사용 검증 미완료** |
| OCR/YOLO 통합 | OCR 기반 시계는 선택적. YOLO11 전체 코칭 근거 종단 결합 | 미완료/미검증 |
| Agent 기반 툴 라우팅, 공식 Riot RSO, 상용 운영 | 공식 인증·에이전트 자동 의사결정·배포 | 미구현 |

## 4. 지금까지 실제 검증한 데이터

### 4.1 Gemini v4 Top-K 장면 탐지

- `LOL_001`: **키아나 정글**, 약 24분 22초. v4 Top-5 추출, 기존 사람 검토 중요도(0~5): **1, 5, 3, 4, 4**.
- `LOL_002`: **카이사 ADC**, 약 15분 22초(조기 서렌). v4 Top-5 추출, 기존 사람 검토 중요도(0~5): **4, 4, 5, 5, 4**.
- 임시 양성 기준 `사람 중요도 >=3`으로 계산한 **선정 결과 Precision@5**는 LOL_001 **4/5(80%)**, LOL_002 **5/5(100%)**, 합산 **9/10(90%)**. **두 VOD에서 사후 평가한 예비 수치일 뿐, 일반화 정확도나 Recall@5가 아니다.** 전체 VOD에 대한 독립 GT 이벤트 목록이 없어 누락 사건 수를 아직 계산할 수 없다.
- 과거 사람 검토 표의 `description factual 0/1`은 **키아나와 카이사의 킬 주체·이동 목적 등에 대한 정정 메모와 모순되는 행이 있으므로 정확도 지표로 직접 사용하지 않는다.** 구체적 주장 단위로 다시 라벨링한다.
- v4의 내부 후보 `score`는 선택용 탐지 점수이지 정확도(%)가 아니다. 별도 논의한 **0~3점 신규 이벤트 루브릭은 다음 버전 제안**이며, 현재 v4.0 운용의 0~5점 중요도 및 사람 GT와 혼동하지 않는다.

### 4.2 Riot API 실경기 결합(LOL_001)

- Match ID **`KR_8440518606`**를 사용자 확인을 통해 LOL_001에 연결. 자동 영상 동일 경기 증명은 아니다.
- 실제 게임 버전 `16.21.834.2754` → Data Dragon 문서 버전 `16.21.1`.
- 키아나 `participantId=7`, 경기 최종 K/D/A **9/3/6**.
- 두 킬을 교차검증: Riot 게임 시각 **10:39, 10:44**, MP4에서 **10:40, 10:45**. 이 두 앵커의 `MP4 − 경기 시각 = +1초`로 일치.
- Top-5 선정 장면에 Riot Timeline 이벤트 총 **27건 연결**. 이는 **킬 27회/코칭 근거 27개라는 뜻이 아니다**. 아이템 구매/파괴·와드 등 비코칭 이벤트가 포함됨.
- 주요 검증 내용: scene_001 **바론 막타가 키아나(killerId 7)**, scene_002 **키아나 직접 킬 2회**, scene_003 **킬 2회·전령 막타**, scene_004 **처치 어시스트 3회·바다 드래곤 막타**, scene_005 **키아나 사망 1회·후속 처치 어시스트 1회**.
- `CHAMPION_KILL`의 킬/사망/어시는 반드시 `killerId/victimId/assistingParticipantIds`로 따로 확인. `participant_or_assist` 문자열만으로 킬 주체를 판정하지 않음.
- Data Dragon의 스킬 설명은 챔피언 능력의 규칙을 설명하는 것. **VOD에서 실제로 스킬을 썼는지, 피했는지 증명하지 않는다.** 종료 아이템은 과거 장면 당시 보유 증거가 아니다.

### 4.3 Riot+RAG 코칭 A/B 실험 결과(LOL_001, n=5쌍)

**입력 조건:** 동일 Top-5 장면의 동일 이미지(장면당 기본 6프레임)와 동일 Gemini 모델. `baseline`은 영상+챔피언만, `assisted`는 동일 영상+Riot 이벤트+패치별 Data Dragon 근거를 제공. 제공된 `LOL_001_RIOT_RAG_AB_RESULTS.zip`에는 두 군 각 5건, 총 10건의 코칭 JSON이 저장됨. 이것은 한 VOD의 파일 단위 비교이고 독립 표본 실험이 아니다.

| 지표(해당 5쌍의 JSON) | Baseline | Assisted | 변화 |
|---|---:|---:|---:|
| 평균 Gemini 응답 지연시간 | **11.16초** | **15.88초** | **+42.3%** |
| 5개 장면 응답시간 합계 | 55.78초 | 79.38초 | +23.59초 |
| 프롬프트 입력 토큰 합계 | 34,890 | 43,282 | +24.1% |
| 출력 토큰 합계 | 2,380 | 3,167 | +33.1% |
| API가 보고한 총 토큰 합계 | 42,608 | 54,353 | +27.6% |
| 생성한 `facts` 개수 | 15 | 20 | 정확도 지표 아님 |

**관찰:** 보조군은 바론·전령·킬·사망의 Riot 사건 시간 및 주체를 더 명시적으로 인용했다. 그러나 일부 코칭은 **시스템 메시지에 키아나 궁극기 사용 기록이 나온다**고 근거 없이 단정했으며, 일부 핵심 어시스트를 누락했다. `facts` 문장의 정확도/근거 지원률은 **프레임과 독립 GT 검토 전까지 미산정**. 추가 입력 토큰·응답시간 증가를 이미 관찰했으며, 종단 소요시간·실제 API 청구액은 미측정.

## 5. 웹 서비스 현재 상태 — 구현과 실제 사용 검증 구분

| 항목 | 현황 |
|---|---|
| 서비스 자체 계정 회원가입/로그인 | 구현; FastAPI 로컬 모의 테스트 통과. **Riot 공식 로그인 아님** |
| Riot ID 입력으로 PUUID 및 최근 대전 기록 조회 | 구현; 계정 소유권 인증 아님 |
| 사용자 선택 경기의 Match/Timeline 조회·영상 길이 1차 검사 | 구현; 동일 경기의 시각/내용 자동 대조는 미완료 |
| MP4 업로드 → 작업큐 → 탐지 → RAG → 코칭 | 서버 측 실행 흐름 구현; 실제 사용자 PC 유료 API 종단 성공 여부 미확인 |
| HUD 시각 자동 인식 | 선택적 Tesseract OCR, 3개 이상 일관된 관측 필요. 실패 시 장면별 Riot 이벤트 연결 보류 |
| 로컬 웹 서버 | Windows PowerShell에서 Uvicorn 시작, `GET /` 200 응답 실제 확인 |
| `GET /api/me` 401 | **로그인 전에 발생한다면 정상** |
| `GET /favicon.ico` 404 | 사이트 아이콘 부재, 코칭 분석과 무관 |
| 인터넷 공개 및 다중 사용자 배포 | **금지/미준비**: 현재는 `127.0.0.1` 로컬 개발용, 단일 워커 큐 |

## 6. 로컬 실행 재현

> 프로젝트 루트에서 PowerShell 사용. `.env`는 공유/커밋하지 말 것.

```powershell
# 의존성 및 테스트
uv pip install -r requirements_service.txt
python -m unittest discover -s tests -p test_auto_service.py -v

# 로컬 웹 시작
python -m uvicorn service.app:app --host 127.0.0.1 --port 8765
# 웹: http://127.0.0.1:8765

# 기존 탐지(실제 API 사용, 비용 발생)
python -m pipeline.cli discover --video videos/LOL_001.mp4 --discovery-id LOL_001_COACH_V4 --champion Qiyana --role jungle --execute

# 기존 Riot+RAG A/B 리플레이(실행 전 프롬프트만 확인)
python -m RAG.cli coach --discovery-id LOL_001_COACH_V4 --model gemini-3.6-flash --run-tag riot_rag_ab02 --ab
# 실제 API 호출: 뒤에 --execute (유료, 새 실행 태그 사용)
```

`service/` 기본 작업은 **assisted 코칭만** 생성하며, 위 `--ab` 10회 실험은 별도로 실행한다. `.env`의 `RIOT_API_KEY`, `GEMINI_API_KEY`를 필요할 때 로드하고, 개발용 Riot 키는 만료될 수 있다. 기본 경기 목록은 최근 **30일·20건**(API 권한·데이터 보관에 따라 다름). 개인 영상, 경기 원본 JSON, 계정 DB는 GitHub에 올리지 않는다.

## 7. 정답 라벨(Ground Truth) 정의

### 7.1 반드시 라벨을 세 층으로 분리

**A. 영상 장면 GT(탐지 평가):** Top-K 결과를 보지 않고 **전체 영상**을 검토해 `event_id`, `video_id`, `start_mp4_sec`, `end_mp4_sec`, `event_type`, `target_player_related`, `human_importance_0_5`, `evidence_time`, `annotator`, `notes`를 기록. 0~5 기준: 0=코칭 가치 없음, 1=배경/단순 파밍·안전 오브젝트, 2=약한 피드백, 3=의미 있는 판단·스킬·교전, 4=중요한 성공/실수, 5=경기 결과/생존/교전 기여가 큰 결정적 상황. **3 이상만 탐지 양성**으로 정의(실험 전에 고정). 킬 외에도 도주·스킬 회피·위치 선정·라인 운영·귀환 판단 등 코칭 가치가 있는 사건을 포함. 반드시 독립적으로 전체 영상을 확인해야 Recall 계산 가능.

**B. 객관적 사건 GT(정합성 평가):** `match_id`, `participantId`, `event_time_game_ms`, `event_time_mp4_sec`, `event_type`, `killerId`, `victimId`, `assistingParticipantIds`, `monsterType`, `role_in_event`, `evidence_type`, `verified` 등을 기록. 실제 Riot 기록에서 킬·사망·어시·오브젝트를 확인하고, 영상과 시간 앵커를 교차검증. **Riot에 없는 QWER 시전·회피·정확한 체력·포지셔닝은 영상으로만 검증**. Match JSON과 실제 영상 경기 동일성은 별도 표시.

**C. 문장/코칭 GT(생성 평가):** 모델 출력의 `facts`, `summary`, `praise`, `improvements`를 **원자적 주장 단위**로 분리하여 사람 평가자가 독립 근거로 채점. 예: “10:39 키아나가 직접 킬”은 한 주장, “키아나가 R을 시전해 킬”은 `R 사용`과 `킬 기여`로 분리. 각 주장에 다음 상태 중 하나를 기입한다.

| 상태 | 정의 | 지표 처리 |
|---|---|---|
| `SUPPORTED` | 독립 Riot 기록·영상 프레임·해당 패치 공식 자료로 직접 뒷받침 | 근거 확인 성공 |
| `CONTRADICTED` | 반대 증거가 확인되는 명백한 오류(예: Bard 킬을 Kai'Sa 킬로 귀속) | 오류로 계산 |
| `UNSUPPORTED` | 주장 자체는 가능하나 모델이 제시한 근거가 **그 사실을 증명하지 못함**(예: 스킬 설명만 보고 R 시전 단정) | 근거지원률 실패; 오류로 단정하지는 않음 |
| `UNKNOWN` | 원본 프레임 가림·샘플 부족 등으로 참/거짓 판단 불가 | 별도 비율 보고; 임의로 정답/오류 처리 금지 |
| `NOT_FACT` | 순수 의견·권고 등 검증 가능한 사실 명제가 아님 | 사실 정확도 분모 제외; 코칭 유용성 별도 평가 |

반드시 `evidence_source(video/riot/patch)`, `evidence_timestamp_or_url`, `correction`, `annotator`를 함께 기록한다. **모델의 `source_type` 자기신고는 정답 라벨이 아니다.** `UNKNOWN` 또는 `UNSUPPORTED`를 자동으로 “거짓”으로 세지 않는다.

### 7.2 평가자 운영

1. **2명 이상 독립 영상 라벨링**: 가능하면 모델 출력을 숨기고 전체 VOD를 보며 GT 사건과 0~5 중요도를 매긴다. 사건의 시작·종료 기준(교전 진입/종료, 회피 행동 구간)을 문서화한다.
2. **분쟁 조정**: 시각 범위가 어긋나면 같은 사건인지 확인하고 한 번만 기록. 중요도 차이가 큰 경우 제3자 또는 합의 기준으로 정리. 원래 라벨 두 개와 최종 확정본을 모두 보존.
3. **Riot 기록은 GT 사건을 검증하는 보조 출처**: 게임 사건 유형에는 쓰되 영상 기반 코칭 맥락·스킬 시전의 정답을 Riot만으로 만들지 않음.
4. **프롬프트/패치/모델 고정**: 평가 영상은 튜닝 데이터와 분리. 프롬프트를 수정한 실험은 새 `run_tag` 및 버전으로 보존.
5. **블라인드 A/B**: 코칭 결과의 Baseline/Assisted 표시를 숨겨 사람 평가자가 0~2점 코칭 유용성(구체성·실행 가능성·균형성·불확실성 표시)을 채점한 뒤 실험군을 공개한다.

## 8. 지표 및 계산식

### 8.1 장면 선별

- **Precision@5** = Top-5 중 *영상 GT 중요도 ≥3*과 매칭된 장면 수 ÷ 5. 선정된 장면 자체의 사람 평가만으로 계산 가능하지만, 사람 라벨이 없으면 계산 불가.
- **Recall@5** = Top-5가 포착한 서로 다른 *전체 VOD GT 중요 사건* 수 ÷ 전체 VOD GT 중요 사건 수. **전체 영상 GT가 있어야만 계산**. 모델의 후보 목록이나 Top-5 자체를 GT로 쓰지 말 것.
- **Event temporal localization** = 동일 사건 여부를 사람에게 확인 후, 사건 중심점 오차(초), 시간대 IoU 등 추가 측정. 한 긴 scene 안에 여러 GT 사건이 있으면 사건별 중복 카운트를 방지하는 1:1 매칭 규칙을 먼저 정의.
- **NDCG@5 / 평균 사람 중요도**(선택) = 상위 랭킹 품질. Precision@5와 별도.

### 8.2 Riot 시간·사건 정합성

- **경기 일치 검증률** = 독립 확인된 Match/VOD 페어 ÷ 제출 페어. 사용자가 리스트에서 경기를 고른 경우에도 ‘사용자 선택’과 ‘자동 입증’을 구분.
- **시간 동기화 오차** = `|예측 MP4 사건 시각 − 사람 확인 MP4 사건 시각|` (평균/중앙값/최댓값). *동일 킬 등 확인된 앵커*로 검증.
- **킬 주체 정확도** = 검증한 킬 주장 중 실제 killerId·victimId·assistingParticipantIds와 일치한 수 ÷ 검증 가능한 킬 주장 수.
- **코칭 유효 사건 연결률** = 유의미한 GT 사건 중 올바른 Riot 기록과 연결된 수 ÷ Riot에 존재하는 해당 GT 사건 수. 구매/와드 로그 등 모든 raw 이벤트 수로 계산하지 않음.

### 8.3 생성된 사실 및 코칭

- **검증 가능 사실 정확도** = `SUPPORTED / (SUPPORTED + CONTRADICTED)`; 별도로 **검증 가능 주장 비율**을 반드시 병기. UNKNOWN·UNSUPPORTED는 여기서 분리.
- **전체 사실 주장 근거지원률** = `SUPPORTED / (SUPPORTED + CONTRADICTED + UNSUPPORTED + UNKNOWN)`; 단 `NOT_FACT` 제외. 이렇게 정의하면 근거가 없는 주장은 낮은 지원률로 나타나지만 “틀린 사실”로 무조건 판정되진 않는다.
- **명백 오류율** = `CONTRADICTED / (SUPPORTED + CONTRADICTED)`; 검증 가능 주장에 한정된 수치임을 표기.
- **근거 부족률** = `UNSUPPORTED / 전체 사실 주장 수`; **판단 불가율** = `UNKNOWN / 전체 사실 주장 수`를 별도 보고.
- **코칭 실용성 0~2점 루브릭**(사람 평가): 상황 이해, 칭찬·비판의 적절성, 행동 지시 구체성, 과신/불확실성 처리 각각 0=부적절, 1=부분 적절, 2=명확·검증 가능. 점수는 *사람의 주관적 유용성*, 자동 정확도가 아니다.

### 8.4 처리시간·비용

- `discover` 전체 영상 분석 시간, Riot 경기 조회 및 캐시 시간, OCR 시간, RAG 검색 시간, 장면별 Gemini 코칭 시간, **영상 업로드부터 최종 결과까지 종단 지연시간**을 각각 로그화.
- Prompt/Output/Total tokens, API 호출 수, 캐시 적중률, 오류/429 재시도 비율, provider 청구액(있으면)을 집계.
- **기존 11.16초/15.88초는 Gemini 코칭 호출의 응답시간이며 VOD 종단 처리시간이 아니다.**

## 9. 다음 실험 설계 — 공정한 비교

**최소 실험:** 같은 VOD, 같은 장면, 같은 프레임, 같은 Gemini 모델/temperature에서 `Baseline(영상만)` 대 `Riot+RAG(근거 제공)`을 비교. 각 생성 주장에 독립 상태 라벨 부여. **현재 LOL_001 5쌍은 이 최소 실험의 예비 결과**이다. 실제 정확도 차이의 확정치가 아님.

**추가 권장 Ablation(아직 미구현/미실행):**

| 실험군 | 영상 | Riot 사건 | 패치 RAG | 목적 |
|---|---|---|---|---|
| A. VLM only | O | X | X | 기준 |
| B. VLM + Riot | O | O | X | 킬·어시·오브젝트의 사실 정합 효과 |
| C. VLM + patch RAG | O | X | O | 스킬/아이템 규칙 정확도 효과 |
| D. VLM + Riot + patch RAG | O | O | O | 통합 이득·추가 비용 |

현실적인 추가 검증: 서로 다른 챔피언/포지션·승패·15분 서렌/긴 경기·해상도·HUD 배치·편집 여부가 다른 **최소 10~20 VOD로 확대하는 탐색적 목표**를 세우고, 시간/예산에 따라 점차 확장. 이는 통계적 신뢰도 보장의 마법 같은 최소 개수가 아니며, 최종 보고서에는 VOD 수·GT 사건 수·주장 수·평가자 수를 모두 기재.

## 10. 현재 미해결 위험 및 다음 우선순위

1. **로그인**: 자체 로컬 계정만 가능. Riot RSO/OAuth2는 별도 승인·구현 필요. Riot 비밀번호를 수집하지 않음.
2. **경기 일치**: 영상 길이·선택 계정 검사는 1차 방어, MP4와 선택한 Match의 **자동 동일 경기 증명은 미구현**.
3. **시간 OCR**: 반복 시계 읽기만으로 안정성 후보를 추정. 이미 편집된 영상·UI 변경에 대해 자동 이벤트 교차검증 추가 필요.
4. **RAG**: `Data Dragon` 패치별 정적 스냅샷 중심; 공식 패치노트 변경 내역/아이템 세부 동적 변수의 수치 검증은 미완성.
5. **QWER와 HUD**: 기존 OpenCV 보조 신호만으로 모든 스킬 시전·회피/적 체력 확정 불가; YOLO11 종단 성능 미검증.
6. **코칭 환각**: `source_type=video`라도 프레임에 없으면 사실로 인정 금지. 화면에 남은 시스템 메시지와 사건 발생 시각 혼동 방지.
7. **서비스 안정성/보안**: 단일 스레드 작업 큐와 SQLite를 사용하는 **로컬 프로토타입**. HTTPS, 보안 점검, 권한·동의, 보관 기간, 업로드 삭제, CSRF·속도 제한, 별도 워커, 장애 복구, API 비용 제한이 필요. 외부 공개 금지.
8. **평가 누락**: 전체 VOD 독립 GT, 미선정 중요 사건 리뷰, 사람 2인 판정, 사실 주장 단위 점수, 기계별 Ablation, 종단 시간·실제 비용이 아직 필요.

## 11. GitHub에 올릴 코드와 제외할 데이터

**추적 권장:** `pipeline/`의 소스, `RIOT/`, `RAG/`, `service/`, `tests/`, `requirements*.txt`, `.gitignore`, `README.md`, 이 상태 문서(`docs/PROJECT_STATUS_AND_EVALUATION.md`), 비밀 값이 없는 `.env.example`.

**추적 금지:** `.env`, `data/riot/`(Match/Timeline/PUUID), `data/service/`(계정·세션 DB), `videos/`(특히 `videos/uploads/`), `results/`(Gemini 응답·스크린샷·로그), 가상환경·모델 가중치·개인 평가 원본. **GitIgnore는 과거 추적/커밋된 비밀을 제거하지 않는다.**

### 관련 개발/평가 명령어

```powershell
# 변경 확인 → 의도한 소스만 스테이징 → 점검
git status
git add .gitignore README.md requirements_service.txt RIOT/ RAG/ service/ tests/
git add docs/PROJECT_STATUS_AND_EVALUATION.md
git diff --cached --name-status
git diff --cached --check

# 모두 확인하고 커밋 / 원격 최신화 / 푸시
git commit -m "feat: add Riot RAG integration and local analysis service"
git pull --rebase origin main
git push origin main
git status
```

