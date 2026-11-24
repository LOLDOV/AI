from pathlib import Path

import cv2

from vod.scenes import time_label
from vod.sync import video_to_game
from vod.video import frame_at

# 장면별 대표 프레임을 JPG로 저장 (VLM 입력·리포트 근거용)

MAX_WIDTH = 1280
JPEG_QUALITY = 88


def export_scene_frames(video_path, scene, sync, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, video_sec in enumerate(scene['frame_times']):
        frame = frame_at(video_path, video_sec)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
        path = out_dir / f'rank{scene["rank"]:02d}_f{idx:02d}_{video_sec:07.1f}s.jpg'
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frames.append({
            'frame_id': f'r{scene["rank"]:02d}f{idx:02d}',
            'path': str(path),
            'video_sec': video_sec,
            'game_time': time_label(video_to_game(sync, video_sec)),
        })
    return frames
