import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# 장면 탐지 평가: 수동 정답 구간과 시간 IoU >= 0.5 로 1:1 매칭해 Precision / Recall 계산.
# 정답 CSV 형식 (영상 시간, 초):
#   video,start_sec,end_sec,type
#   LOL_002,354,372,teamfight
#   LOL_002,610,628,event_free_mistake
# type 이 'event_free' 로 시작하면 이벤트 없는 장면으로 별도 집계한다.
#
# 예) python -m evaluation.scene_eval labels/scenes.csv --top-k 10

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / 'results'
IOU_THRESHOLD = 0.5


def temporal_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(preds, labels, threshold=IOU_THRESHOLD):
    pairs = sorted(
        ((temporal_iou(p, l), i, j) for i, p in enumerate(preds) for j, l in enumerate(labels)),
        reverse=True,
    )
    used_p, used_l, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < threshold:
            break
        if i in used_p or j in used_l:
            continue
        used_p.add(i)
        used_l.add(j)
        matched.append((i, j, iou))
    return matched


def load_labels(path):
    labels = defaultdict(list)
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            labels[row['video']].append({
                'start': float(row['start_sec']),
                'end': float(row['end_sec']),
                'type': row.get('type', '') or 'unknown',
            })
    return labels


def evaluate(label_path, top_k=None):
    labels = load_labels(label_path)
    total_tp = total_pred = total_label = 0
    by_type = defaultdict(lambda: [0, 0])  # type -> [matched, total]
    per_video = {}

    for video, video_labels in labels.items():
        scenes_path = RESULT_DIR / video / 'scenes.json'
        if not scenes_path.is_file():
            print(f'skip {video}: no scenes.json')
            continue
        with scenes_path.open(encoding='utf-8') as f:
            scenes = json.load(f)['all_scenes']
        if top_k:
            scenes = sorted(scenes, key=lambda s: s['rank'])[:top_k]

        preds = [(s['start'], s['end']) for s in scenes]
        gts = [(l['start'], l['end']) for l in video_labels]
        matched = match(preds, gts)
        matched_labels = {j for _, j, _ in matched}

        for j, label in enumerate(video_labels):
            by_type[label['type']][1] += 1
            if j in matched_labels:
                by_type[label['type']][0] += 1

        total_tp += len(matched)
        total_pred += len(preds)
        total_label += len(gts)
        per_video[video] = {
            'tp': len(matched), 'pred': len(preds), 'label': len(gts),
            'mean_iou': round(sum(m[2] for m in matched) / len(matched), 3) if matched else None,
        }

    result = {
        'iou_threshold': IOU_THRESHOLD,
        'top_k': top_k,
        'precision': round(total_tp / total_pred, 3) if total_pred else None,
        'recall': round(total_tp / total_label, 3) if total_label else None,
        'counts': {'tp': total_tp, 'pred': total_pred, 'label': total_label},
        'recall_by_type': {t: {'matched': m, 'total': n, 'recall': round(m / n, 3)} for t, (m, n) in by_type.items()},
        'per_video': per_video,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', help='ground-truth CSV')
    parser.add_argument('--top-k', type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.labels, args.top_k), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
