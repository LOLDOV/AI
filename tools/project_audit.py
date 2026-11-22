"""Non-destructive inventory of legacy source & results trees. No network, edits or deletion.

Run: python tools/project_audit.py
"""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'results' / 'project_audit' / 'refactor_inventory.md'
LEGACY = ('OpenCV','gemini','OCR','rag','qwen')
RESULT_FOLDERS = {
    'combat_segments_eval_v2': '기존 검증 보고서와 연결된 프레임: 우선 보존',
    'hud_temporal': '수작업 스킬/CC 정답 및 원본 크롭: 반드시 보존',
    'hud_coaching_evidence': '코칭 검증 근거: 보존',
    'gemini_episodes': '기존 Gemini 원문, 비교 실험 기준: 보존',
    'summoner_spells': 'D/F 및 강타 진화 판독 원본: 보존',
    'hp_ocr': 'HP v1 실패 기록도 성능 평가에 필요: 보존',
    'hp_ocr_v2': 'HP v2 실패 기록: 보존',
    'hp_paddle_benchmark': 'Paddle v3 비교 기록: 보존',
    'combat_segments': '기존 장면 클립. 새 버전 검증 후 보관 후보',
    'extracted_frames': '초기 후보 프레임. 새 버전 검증 후 보관 후보',
    'gemini_candidates': '과거 후보별 성공/실패 API 로그: 재현성 위해 우선 보존',
    'hud_analysis': '초기 HUD 실험: 수작업/검증 포함 여부 확인 전 삭제 금지',
    'qiyana_event_candidates': '초기 키아나 전용 후보 산출물: 보관 후보',
}


def digest(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def run(dest:Path=REPORT):
    rows=['# LOLDOV 프로젝트 실제 로컬 현황 (삭제/수정 없는 점검)', '',
          f'- 생성: {datetime.now().isoformat(timespec="seconds")}',
          '- **결과는 현재 이 컴퓨터에서 발견된 파일 기준. 파일 이동/삭제 없음.**',
          '', '## 소스 파일 현황', '', '| 영역 | Python 개수 | 참고 |','|---|---:|---|']
    sha_to_paths={}
    for name in (*LEGACY,'pipeline','evaluation'):
        d=ROOT/name
        files=sorted(d.rglob('*.py')) if d.exists() else []
        rows.append(f'| `{name}/` | {len(files)} | {"미발견" if not d.exists() else "기존 위치 보존"} |')
        for f in files:
            sha_to_paths.setdefault(digest(f),[]).append(f.relative_to(ROOT).as_posix())
    dup=[paths for paths in sha_to_paths.values() if len(paths)>1]
    rows.extend(['','## 동일한 내용의 Python 파일 (SHA-256 일치)',''])
    if dup:
        rows.extend(['- '+', '.join(f'`{name}`' for name in group) for group in dup])
    else:
        rows.append('- 동일 바이트 Python 중복 없음 (기능 중복 여부는 별도 검토)')
    rows.extend(['','## 결과 폴더 정책 (존재 여부 확인)','', '| 폴더 | 상태 | 권장 처리 |','|---|---|---|'])
    for name,desc in RESULT_FOLDERS.items():
        d=ROOT/'results'/name
        rows.append(f'| `results/{name}` | {"있음" if d.exists() else "없음"} | {desc} |')
    rows.extend(['','## 새 실행 체계', '',
        '- `results/pipeline/<VIDEO_ID>/` 안에 새로운 실험이 독립 저장됨',
        '- 코드 교체 전/후의 결과는 별도 디렉터리로 보존',
        '- `qwen/`은 아직 소스 미검토 상태: **자동 정리 제외**',
        '- `data/`, `videos/`, `results/`는 `.gitignore` 상태 확인',
        '- 서로 다른 이름의 결과가 실제로 같은 데이터인지 확정하려면 파일·해시 비교가 필요함',
        '- 이 도구는 **삭제, 복사, 이동, rename 하지 않음**'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text('\n'.join(rows)+'\n',encoding='utf8')
    return dest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPORT)
    args=parser.parse_args()
    print('READ_ONLY_AUDIT:',run(args.out))
