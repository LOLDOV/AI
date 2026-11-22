# 이전·폴더 정리 체크리스트 (Windows)

- [ ] 1. `git status --short`를 저장하고, Git 미추적(`??`) Python 파일까지 별도 백업
- [ ] 2. 실제 소스 폴더 `OpenCV`, `gemini`, `OCR`, `rag`, `qwen`을 바깥 폴더에 복사하여 백업
- [ ] 3. v2 ZIP에서 `pipeline`, `configs`, `evaluation`, `tests`, `docs`, `tools`, `OCR/hp_paddle_benchmark.py` 복사
- [ ] 4. 기존 `OpenCV`·`gemini`·`rag`·`qwen`의 원본 파일은 아직 삭제하지 않음
- [ ] 5. `python -m pip install -r requirements_pipeline.txt`
- [ ] 6. `python -m unittest discover -s tests -v` 통과
- [ ] 7. `python tools/project_audit.py` 실행하여 파일별 중복 확인
- [ ] 8. `LOL_001_V2` 이름으로 prepare/analyze/run(드라이런) 수행
- [ ] 9. `results/pipeline/LOL_001_V2/manifest.json`, `machine_evidence.json` 확인
- [ ] 10. 자동 후보가 사람 정답처럼 전달되지 않았는지 `assisted_prompt.txt` 확인
- [ ] 11. Gemini API가 유효하면 --execute (비용 주의), 양쪽 output 확인
- [ ] 12. `LOL_002` 또는 별도 VOD로 프로파일 검증 후 다시 실행
- [ ] 13. `docs/LEGACY_SOURCE_MAP.csv` 중 `ARCHIVE_UNIQUE` 파일을 대체할 기능이 없는지 확인
- [ ] 14. 다른 코드의 `from OpenCV`, `import OpenCV`, `gemini/`, `qwen/` 경로 의존성 확인
- [ ] 15. 기존 결과/코드의 최종 백업 검증 후, 그제야 수동으로 아카이브로 이동 검토

## 지워서는 안 되는 것

- `results/hud_temporal/LOL_001/**/hud_manual_validation.csv` — 정답 라벨 포함
- `results/hud_temporal/LOL_001/**/hud_cc_context.json` — 사람 확인 기록 포함 가능
- `results/combat_segments_eval_v2/LOL_001/` — 기존 타임라인 평가 근거
- `results/gemini_episodes/LOL_001/` — 기존 보고서 원문
- `results/summoner_spells/LOL_001/` — D/F 템플릿 및 RAG 교정 이력
- `results/hp_ocr*`, `results/hp_paddle_benchmark` — 성능 비교 재현
- `videos/`, `data/riot/`, `.env`, `.venv/`, `qwen/` — 자동 삭제 금지

## Git 올리기 전 점검

```powershell
git status --short
git check-ignore .env videos/LOL_001.mp4 results/pipeline/LOL_001_V2/manifest.json
```

`.gitignore`에 아직 등록되지 않았다면 `.gitignore.pipeline.snippet`의 항목을 **검토 후 기존 ignore에 병합**하세요.
`.env`를 업로드하지 말고, 개인/타인 VOD와 Riot 매치 JSON도 공개 레포지토리에 올리지 마세요.
