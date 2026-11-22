"""Read-only inventory to plan a safe folder migration; does not move files."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from pathlib import Path
from .core import ROOT

SKIP={'.git','.venv','venv','__pycache__','node_modules','.pytest_cache','results','.idea'}
CATEGORIES={
 'OpenCV':'기존 이벤트 탐지·프레임/HUD 전처리 (현위치 보존)',
 'OCR':'HP OCR 실험용 (현위치 보존)',
 'gemini':'기존 VLM 실험 및 코칭 (현위치 보존)',
 'rag':'게임 규칙 지식 기반 (현위치 보존)',
 'qwen':'이전 VLM 후보 모델 실험 (현위치 보존)',
 'data':'Riot API/데이터 (Git 제외)',
 'videos':'원본 VOD 영상 (Git 제외)',
 'pipeline':'NEW: 프로젝트 통합 실행기',
 'configs':'NEW: HUD 좌표 프로파일',
 'tests':'NEW: 회귀/누출 방지 테스트',
 'results':'분석 출력 (Git 제외)'
}


def inventory():
    rows=['# LOLDOV 기존 폴더 감사 (읽기 전용)','',
          f'- 생성 시각: {datetime.now().isoformat(timespec="seconds")}',
          '- 이 명령은 **기존 파일 이동·삭제·이름변경을 하지 않습니다.**',
          '- 상대 경로에 의존하는 기존 스크립트들은 당분간 현위치에 두세요.','',
          '| 폴더 | 의미 | Python 파일 수(상위 일부) |', '|---|---|---:|']
    for name,desc in CATEGORIES.items():
        d=ROOT/name
        n=len(list(d.rglob('*.py'))) if d.is_dir() and name not in {'data','videos','results'} else 0
        rows.append(f'| `{name}/` | {desc} | {n} |')
    rows+=['','## 프로젝트 루트의 Python 파일 (기존은 바로 이동 금지)','']
    roots=sorted(p.name for p in ROOT.glob('*.py'))
    rows+=['- '+r for r in roots] or ['- 루트 파일 없음']
    rows+=['','## 향후 구조 정리 권장','',
           '1. 신규 코드는 `pipeline/`과 기존 기능 폴더를 통해 확장',
           '2. 데이터 경로는 `pipeline/core.py`의 프로젝트 ROOT로 통일',
           '3. 수작업 정답은 평가 단계에만 사용하고 모델 입력 경로에서 분리',
           '4. 각 기능의 테스트 추가 후 개별 이동 (한꺼번에 이동하지 않기)',
           '5. 다음 주요 리팩터링은 기존 `OpenCV/` 파일의 경로 하드코딩 제거',
           '','## Git 주의','',
           '- `.env`, `videos/`, `data/riot/`, `results/`, `.venv/` 업로드 금지',
           '- `git status --short`로 사전 확인; 원본 플레이/지인 개인정보 포함 파일은 공개 저장소 업로드 금지',
           '- 예시 ignore 설정은 `.gitignore.pipeline.snippet` 참조']
    dest=ROOT/'results'/'project_audit'/'folder_inventory.md'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    return dest
