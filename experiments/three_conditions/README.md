# LOLDOV: 3-condition fair death-coaching ablation

> Tested against the **uploaded full source snapshot dated 2026-11-28**. The deployed service and existing `link/run.py` are **not modified**. Python unit tests cover local mocked behavior; **the 4 real videos and paid Gemini API were not executed here**.

## What is tested

| Condition | Same 16-second Riot-death scene + 8 identical frames? | Riot facts as Gemini prompt | Patch RAG as Gemini prompt | Post-generation claim check |
|---|---|---|---|---|
| `full` | yes | yes | yes | yes |
| `no_verify` | yes | yes | yes | **no** |
| `vlm_only` | yes | **no** | **no** | **no** |

**Critical controls**

- Riot Match/Timeline are used **only to select and time-align the common death windows**. This does **not** mean that `vlm_only` can find deaths by itself: it is *VLM-only coaching on externally selected scenes*.
- `full` and `no_verify` use the **same single Gemini generated response**; the difference is whether `link.verify` holds unsupported/conflicting claims. This isolates the checker without Gemini sampling variance. No Verify does not call `verify.verify()` a second time.
- The checker currently checks **ref IDs and a few contradictions**, not actual video truth. Human review is the factual GT. Never report `link.verify`'s `supported` as correctness.
- OCR alignment is computed **once per video and reused for all three conditions**. If you use a manual offset fallback, it must be explicitly authorized and tagged `manual_non_ocr`; those runs **do not measure OCR**.
- Champion is passed as common metadata to all conditions. Actual Match ID, killer, death details, Riot records, OCR clock, and patch docs are **NOT passed to Gemini** in `vlm_only`.

## Install

Unzip the package in **the project root**, e.g. `C:\Users\by090\Desktop\vlm-project`, retaining:

```text
vlm-project/
├─ experiments/
│  ├─ __init__.py
│  └─ three_conditions/
│     ├─ __init__.py
│     ├─ runner.py
│     ├─ gt_videos_TEMPLATE.csv
│     └─ README.md
├─ tests/
│  └─ test_three_conditions.py
├─ link/     # already exists, unchanged
├─ RIOT/     # already exists, unchanged
├─ RAG/      # already exists, unchanged
├─ vod/      # already exists, unchanged
└─ videos/   # local only, not shared to GitHub
```

Use the existing project venv with the dependencies already needed for the web pipeline: `cv2`, `rapidocr-onnxruntime`, `google-genai`, `requests`, `python-dotenv`, FFmpeg. Do **not** commit API keys or raw VOD.

## Step 1. Create `gt_videos.csv`

Copy the template to the **project root**:

```powershell
Copy-Item .\experiments\three_conditions\gt_videos_TEMPLATE.csv .\gt_videos.csv
```

Edit it using Excel (save as CSV UTF-8) or VS Code:

```csv
video_id,match_id,champion,riot_id,puuid,video_path,sync_offset_sec
LOL_033,KR_REAL_ID_033,ActualChampion,,,videos/LOL_033.mp4,
LOL_035,KR_REAL_ID_035,ActualChampion,,,videos/LOL_035.mp4,
LOL_039,KR_REAL_ID_039,ActualChampion,,,videos/LOL_039.mp4,
LOL_040,KR_REAL_ID_040,ActualChampion,,,videos/LOL_040.mp4,
```

**These example IDs/champion text are placeholders and will not run**. Fill with the actual Match IDs (e.g. `KR_8440518606`) and official English `championName` such as `Qiyana`, `Kaisa`, `Ahri`. The optional `riot_id` (`name#tag`) or `puuid` prevents confusion with another participant. At least one is recommended. `video_path` can be an absolute path if videos reside under the web uploads folder; paths are resolved against the project root.

`gt_videos.csv` is *video metadata*, not human truth labels. It cannot produce model accuracy alone. The video/match identity is not proven by champion alone: use the actual selected match and confirm visual events/duration before interpreting results.

To skip OCR **only if it fails** or to use known manual alignment, enter the offset in seconds (`video_time - game_time`, e.g. `1.0`) and add `--allow-manual-sync`. This is a **non-OCR** run and must not be presented as OCR-based Full system evaluation. Edited/pause-heavy videos need segment-based OCR alignment; one constant manual offset may be wrong.

## Step 2. Run local tests

```powershell
python -m unittest discover -s tests -p test_three_conditions.py -v
```

## Step 3. Pilot **without paid Gemini**

Run one death from one video, inspect OCR and scene mapping:

```powershell
python -m experiments.three_conditions.runner run --table .\gt_videos.csv --videos LOL_033 --limit-deaths 1 --run-tag pilot_033
```

This command creates **8 common frames** for the first death, loads cached Riot Match/Timeline files, scans the local video with RapidOCR, and loads patch-matched Data Dragon docs; it **does not call Gemini**. It can take time for OCR scanning and can retrieve public Data Dragon data. Add `--fetch` only if the Riot JSON cache is missing and you have a valid key.

Examine:

```
results/ablation_3/pilot_033/LOL_033/scenes.json
results/ablation_3/pilot_033/LOL_033/frames/death_01/
results/ablation_3/pilot_033/LOL_033/experiment_manifest.json
```

Check that `death_01` really shows this player's death (not an unrelated match/camera view) and that OCR time alignment is sensible **before spending Gemini tokens**.

Then execute this one death:

```powershell
python -m experiments.three_conditions.runner run --table .\gt_videos.csv --videos LOL_033 --limit-deaths 1 --run-tag pilot_033 --execute
```

**Expected newly paid Gemini requests: 2 per death**, not 3, because `full` and `no_verify` share the assisted generation. Repeat the command to resume; completed generations should load from cache. If `429` occurs, the tool stops and retains completed results. The actual API bill must be checked with the provider.

## Step 4. Run all 4 videos with their own run tag

```powershell
python -m experiments.three_conditions.runner run --table .\gt_videos.csv --run-tag study_3cond_01 --continue-on-error
```

Inspect all `scenes.json` and shared frames. Then:

```powershell
python -m experiments.three_conditions.runner run --table .\gt_videos.csv --run-tag study_3cond_01 --continue-on-error --execute
```

If local Match/Timeline JSON is absent, include `--fetch`. **Never mix `--limit-deaths 1` pilot outputs with all-death runs using the same run tag.** If inputs, model, frames or source versions change, choose a new `--run-tag` instead of overwriting results.

Results:

```
results/ablation_3/study_3cond_01/
├─ batch_status.json
├─ LOL_033/
│  ├─ experiment_manifest.json
│  ├─ scenes.json
│  ├─ hud_scan.json
│  ├─ frames/death_01/frame_000.jpg ...
│  ├─ generation/death_01/assisted.json
│  ├─ generation/death_01/vlm_only.json
│  ├─ full/death_01.json
│  ├─ no_verify/death_01.json
│  ├─ vlm_only/death_01.json
│  └─ summary.json
├─ LOL_035/
├─ LOL_039/
└─ LOL_040/
```

Inspect `summary.json` for `complete: true` in **all 3 conditions for all 4 videos**. A cached `generation_latency_sec` is the historical Gemini generation latency; it is **not** fresh execution time. `no_verify` reuses Full's generation; **never sum Full + No Verify token counts as two distinct paid requests**.

## Step 5. Generate blind human rating sheets

```powershell
python -m experiments.three_conditions.runner sheet --run-tag study_3cond_01
```

Creates:

```
results/ablation_3/study_3cond_01/human_review/
├─ coaching_rating.csv
├─ claim_review.csv
└─ rating_key_DO_NOT_SHOW_RATERS.json
```

Only the first two CSVs go to the raters. Do not send the key. The output may still differ in style (and be inferable), so full blinding is not guaranteed.

`coaching_rating.csv`: fill **all four** integer fields from 0 to 2: `situation_0_2`, `cause_0_2`, `actionability_0_2`, `evidence_0_2` (total /8). Each evaluator should ideally receive a separate copy.

`claim_review.csv`: every claim, including claims **held/removed by Full's checker**, gets a human label:

- `SUPPORTED`: corroborated by the original video or objectively correct Riot event.
- `CONTRADICTED`: clear contrary evidence.
- `UNSUPPORTED`: cited evidence cannot justify the assertion, but there is no demonstrated contradiction.
- `UNKNOWN`: insufficient evidence to judge.
- `NOT_FACT`: nonfactual advice/opinion.

Provide a video timestamp / Riot event / patch text in `evidence_ref`. Note: `claim_review.csv` labels the structured `claims` array; narrative details embedded in `title`, `situation`, `cause` or `alternatives` need separate atomic-claim labeling if the study intends a **complete coaching factual accuracy** estimate.

The same factual claim can appear in both `full` and `no_verify`. For consistency, identical claims should receive identical labels even though Full may hide them.

## Step 6. Aggregate ratings (no Gemini calls)

```powershell
python -m experiments.three_conditions.runner score --run-tag study_3cond_01
```

Writes `human_scores.csv`, `human_scores.json`. Key metrics:

- `displayed_fact_accuracy_verified`: human-supported / (human-supported + human-contradicted) **among shown claims**.
- `displayed_explicit_error_rate_verified`: human-contradicted / (human-supported + human-contradicted) **among shown claims**.
- `displayed_support_rate_all_factual`: human-supported / all factual shown claims, including unknown/unsupported.
- `correct_claims_held_by_verifier`: human-supported claims that Full's rule-based checker withheld (an **overblocking** signal).
- `coaching_avg_0_8`: human usefulness/quality score (average across rated items).

`UNSUPPORTED` and `UNKNOWN` are not automatically treated as false. Only completely rated conditions are eligible for final comparative conclusions; the output includes completion flags.

## Scope and interpretation

- **This package evaluates death-scene coaching** only. It does not evaluate general Top-K discovery, non-death coaching, or Precision@5/Recall@5. Those require separate scene labels and a discovery experiment.
- **Full vs No Verify** isolates the claim post-filter because it shares the same generation, frames, Riot facts and documents. The checker is an ID/rule audit, not VLM truth assessment.
- **Full vs VLM Only** compares external Riot+RAG+verification as a bundle, not the isolated effect of RAG or Riot alone. Later `no_rag`/`no_ocr` trials are needed for per-component attribution.
- On-screen clock alignment may fail, or non-gameplay cameras/edits may show a different moment. If an event isn't in the recorded video, it is explicitly skipped and the denominator changes; report skip rates.
- Four VODs are a small sample. Report **per-video results**, not only one pooled mean. Avoid putting raw JSON, Match IDs, PUUID, login DB or video in GitHub.
