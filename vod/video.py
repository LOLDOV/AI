import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

# 영상 메타데이터 조회와 프레임 읽기 (ffmpeg 파이프로 일정 간격 샘플링)


def probe(path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,codec_name:format=duration,size',
         '-of', 'json', str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    num, den = stream['r_frame_rate'].split('/')
    return {
        'file': Path(path).name,
        'width': int(stream['width']),
        'height': int(stream['height']),
        'fps': round(int(num) / max(1, int(den)), 3),
        'codec': stream.get('codec_name'),
        'duration_sec': round(float(data['format']['duration']), 2),
        'size_mb': round(int(data['format']['size']) / 1024 / 1024, 1),
    }


def iter_frames(path, sample_fps=1.0, start_sec=0.0, end_sec=None):
    """Yield (video_sec, BGR frame) at a fixed sampling rate."""
    info = probe(path)
    w, h = info['width'], info['height']
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(start_sec), '-i', str(path)]
    if end_sec is not None:
        cmd += ['-t', str(end_sec - start_sec)]
    cmd += ['-vf', f'fps={sample_fps}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']

    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            yield start_sec + index / sample_fps, frame
            index += 1
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def frame_at(path, video_sec):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, video_sec * 1000)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()
