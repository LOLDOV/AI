import cv2
import csv
import os

# 설정

VIDEO_PATH = "1.mp4"

START_TIME = "7.45" # 시작: 7분 45초
END_TIME = "8.20" # 종료: 8분 20초

FRAME_INTERVAL = 1

OUTPUT_DIR = "Qwen_Test"
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")

os.makedirs(FRAME_DIR, exist_ok=True)

# 분.초 > 초
def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split("."))

    if seconds >= 60:
        raise ValueError("초는 60 미만으로 입력해야 합니다.")

    return minutes * 60 + seconds

START_SEC = time_to_seconds(START_TIME)
END_SEC = time_to_seconds(END_TIME)

if END_SEC <= START_SEC:
    raise ValueError("종료 시간은 시작 시간보다 커야 합니다.")

# 원본 영상 열기

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened() or fps <= 0:
    raise RuntimeError("영상을 열 수 없습니다.")

timestamps = []
frame_count = 0

current_sec = START_SEC

# 지정 구간 프레임 추출

while current_sec < END_SEC:

    cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)

    ret, frame = cap.read()

    if not ret:
        print(f"{current_sec}초 프레임 추출 실패")
        current_sec += FRAME_INTERVAL
        continue

    filename = f"frame_{frame_count:03d}.jpg"

    save_path = os.path.join(FRAME_DIR, filename)

    cv2.imwrite(save_path, frame)

    timestamps.append([
        filename,
        current_sec - START_SEC,
        current_sec
    ])

    frame_count += 1
    current_sec += FRAME_INTERVAL

cap.release()

# CSV 저장

with open(
    os.path.join(OUTPUT_DIR, "timestamps.csv"),
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame_file",
        "clip_timestamp_sec",
        "source_timestamp_sec"
    ])

    writer.writerows(timestamps)

print("프레임 추출 완료!")
print(f"추출된 이미지: {frame_count}장")