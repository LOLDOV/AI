# 결과 폴더 정책 — 현재 제공된 `results_tree.txt` 기준

- **제공된 자료에는 폴더명만 있고 개별 파일명·파일 크기·해시는 없습니다.** 두 폴더가 데이터 중복이라고 확정하거나 삭제 대상을 정할 수 없습니다.
- 따라서 **현재 결과 13개 범주 모두 유지**. 이 문서의 「보관 후보」는 삭제 허가가 아닙니다.
- 새 분석 결과는 기존 디렉터리와 별개로 `results/pipeline/<VIDEO_ID>/`에 저장합니다.

| 기존 results 하위 폴더 | 정책 |
|---|---|
| `combat_segments/` | 초기 원본 장면 클립/후보: 보관 후보. 동등성 확인 전 삭제 금지 |
| `combat_segments_eval_v2/` | 11프레임 타임라인 검증과 연결된 영상 장면 이미지: 반드시 보존 |
| `extracted_frames/` | 기존 교전 후보 1~9의 추출물: 기존 연구 재현용 보관 |
| `gemini_candidates/` | Gemini 성공/실패 원문과 모델 비교 기록: 보존 |
| `gemini_episodes/` | 기존 원문 보고서/평가 A·B 비교 기준: 보존 |
| `hp_ocr/` | Tesseract v1 성능 실패 데이터: 평가 자료로 보존 |
| `hp_ocr_v2/` | Tesseract v2 성능 실패 데이터: 평가 자료로 보존 |
| `hp_paddle_benchmark/` | PaddleOCR v3 실험 데이터: 평가 자료로 보존 |
| `hud_analysis/` | 기존 HUD 실험 산출물/수동 검토 중복 가능: 파일 단위 확인 전 보존 |
| `hud_coaching_evidence/` | 사람 검증한 HUD·CC 통합 원문: 보존; 자동 A/B에는 공급 금지 |
| `hud_temporal/` | 수작업 스킬/CC 라벨과 원본 크롭: 반드시 보존; 자동 A/B에는 공급 금지 |
| `qiyana_event_candidates/` | 선별 초기 교전 후보 클립: 보관 후보. 복제 증거 필요 |
| `summoner_spells/` | 공식 아이콘 캐시, D/F 자동 판독 원문 및 파란 강타 수동 보정: 보존 |

## 새 폴더 구조

```text
results/
  pipeline/
    LOL_001/
      manifest.json
      frames/<scene_id>/
      hud_temporal/<scene_id>/
        hud_timeline.json
        hud_state_candidates.json
        hud_state_candidates_with_levels.json  # 선택
        hud_cc_context.json
      summoner_spells/
        summoner_spell_probe.json  # 선택
      machine_evidence.json
      experiments/<run_tag>/
        baseline_prompt.txt
        assisted_prompt.txt
        baseline_result.json              # Gemini 실행 후
        assisted_result.json              # Gemini 실행 후
        comparison_template.csv
```

## 현존 results 대청소 순서 (권장: 삭제하지 않기)

1. `python tools/project_audit.py`로 실제 PC 기준 로컬 목록을 생성한다.
2. 별도의 저장 장치에 `results/`의 중요한 폴더를 백업한다. 특히 `hud_temporal`, `gemini_episodes`, `combat_segments_eval_v2` 필수.
3. 신형 pipeline 테스트 후 구형 파일과 새 파일의 용도/입력을 비교한다.
4. 다수 영상에서 회귀 테스트한 뒤에만 비핵심 프레임 복사본을 보관 폴더로 이동한다.
5. `results/` 자체는 `.gitignore`에 두고 Git 업로드 대상에서 제외한다.

## Git & 개인정보

- Riot API JSON과 타인 VOD는 당사자 동의 범위 안에서 관리. 공개 레포지토리에 업로드하지 않기.
- `.env`, API 키, 원본 MP4, `results/`는 절대 새 ZIP에 동봉하지 않았습니다.
