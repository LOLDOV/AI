유저 정보, 경기 데이터, 용량이 큰 파일 등을 포함하고 있기 때문에 Data, Videos, result 폴더 Git 하지 않음 (구글 드라이브로 공유할 것.)

## LOLDOV 개발 현황 (2026.11.24)

**현재 단계: Gemini v4.0 장면 탐지 + Riot API + 패치별 Data Dragon RAG + 로컬 웹 서비스 MVP**

LOLDOV: League of Legends 경기 VOD를 분석하여 코칭 가치가 있는 주요 장면을 자동 선정하고, Riot API 경기 기록과 실제 경기 패치의 챔피언·아이템 정보를 결합해 코칭 리포트를 생성하는 서비스.

### 1. 시스템 파이프라인

```text
LOLDOV 자체 계정 로그인
        ↓
Riot ID 최초 등록
        ↓
Riot API 대전 기록 조회
        ↓
사용자가 분석할 경기 선택
        ↓
MP4 영상 업로드
        ↓
Gemini v4.0 장면 탐지
Scout → Verify → Top-5
        ↓
Riot Match / Timeline 연결
        ↓
HUD 기반 시간 동기화 시도
        ↓
경기 패치별 Data Dragon RAG
        ↓
Gemini 코칭 리포트 생성
```

현재 로그인은 LOLDOV 자체 계정 기반이며, Riot 공식 로그인(RSO)은 구현되지 않았음. 사용자가 대전 기록에서 경기를 선택하므로 정확한 Match ID를 직접 입력할 필요는 없지만, 영상만으로 동일 경기를 완전히 자동 식별하는 단계는 아직 구현되지 않음.


### 2. 주요 모듈

| 모듈 | 역할 |
|---|---|
| `pipeline/` | OpenCV 프레임 추출, Gemini Scout/Verify, Top-K 장면 선정 |
| `RIOT/` | Riot ID 조회, Match/Timeline 수집, 경기 확인, 시간 동기화 |
| `RAG/` | 해당 경기 패치의 챔피언 스킬·아이템 검색 및 코칭 근거 결합 |
| `service/` | 로컬 로그인, 대전 기록 조회, MP4 업로드, 자동 분석 작업 실행 |
| `tests/` | 모듈 및 웹 서비스 모의 테스트 |


### 3. 개발 및 검증 현황

- Gemini v4.0 기반 전체 VOD 탐지 및 Top-5 장면 선정 기능 구현
- LOL_001(키아나 정글), LOL_002(카이사 ADC) 대상 장면 선정 실험
- LOL_001 실제 Riot Match/Timeline 데이터 연결 및 교차검증
- MP4와 Riot Timeline의 시각 차이 확인
- 선정된 5개 장면에 Riot 이벤트 27건 연결
- 실제 경기 버전 16.21 계열의 Data Dragon 정보 검색
- LOL_001에서 VLM 단독과 Riot+RAG 보조 코칭 A/B 실험 완료
- 로컬 웹 서비스 구현 및 자동화 모의 테스트 통과
- 사용자 Windows 환경에서 웹 서버 실행 및 초기 페이지 접속 확인

**미완료:** Riot 공식 계정 인증, 영상만으로 Match ID 자동 확정, OCR/YOLO 전체 통합 검증, 독립 Ground Truth 기반 Recall@5 및 코칭 정확도 평가, 상용 배포.


### 4. 예비 성능 결과

**장면 선정**

기존 2개 VOD에서 사람이 매긴 중요도 0~5 중 3점 이상인 장면은 Top-5 총 10건 중 9건입니다.

- LOL_001 Precision@5: 80%
- LOL_002 Precision@5: 100%
- 합산 예비 Precision@5: 90%

이 값은 소수 샘플의 선정 장면 관련성 평가임. 모델 전체 정확도나 Recall@5가 아님.

**Riot+RAG 코칭 A/B**

| 지표 | VLM 단독 | Riot+RAG |
|---|---:|---:|
| 평균 Gemini 응답시간 | 11.16초 | 15.88초 |
| 입력 토큰 합계 | 34,890 | 43,282 |
| 출력 토큰 합계 | 2,380 | 3,167 |
| 총 보고 토큰 | 42,608 | 54,353 |

Riot+RAG는 킬·오브젝트의 실제 기록과 발생 시각을 명시적으로 제공하는 데 도움이 되었으나, 일부 스킬 사용 설명은 근거 검증이 필요함. 전체 코칭 정확도 개선은 아직 정량적으로 입증되지 않음.


### 5. Ground Truth 및 평가 예시

장면 탐지, 객관적 사건, 코칭 사실 주장을 분리해 평가.

- **장면 탐지 GT:** 전체 VOD를 독립 검토하여 주요 사건 구간과 중요도(0~5)를 기록. 중요도 3 이상을 양성으로 정의하여 Precision@5, Recall@5 계산.
- **Riot 사건 GT:** 킬·사망·어시스트·오브젝트의 실제 주체와 시각을 Riot Match/Timeline 및 영상으로 교차확인.
- **코칭 주장 GT:** 각 주장을 `SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`, `UNKNOWN`, `NOT_FACT`로 분류하고 독립 근거를 기록.
- **코칭 유용성:** 상황 이해, 행동 지시의 구체성, 칭찬·개선점 균형, 불확실성 처리 능력을 사람이 평가.
- **효율성:** 장면 탐지·Riot 조회·RAG 검색·Gemini 코칭·전체 처리시간, 토큰, API 비용 별도 측정.


### 6. 로컬 실행

```powershell
uv pip install -r requirements_service.txt
python -m unittest discover -s tests -p test_auto_service.py -v
python -m uvicorn service.app:app --host 127.0.0.1 --port 8765
```

웹 접속: `http://127.0.0.1:8765`

로컬 개발용 서비스


### 7. 상세 문서

구현 현황, 실제 검증 결과, 평가 계산식, 정답 라벨링 기준, 실험군 비교, 향후 개선 과제는 [프로젝트 상세 현황 및 평가 프로토콜](docs/PROJECT_STATUS_AND_EVALUATION.md)을 참고.