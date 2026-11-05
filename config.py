import os
from dotenv import load_dotenv

#API 접속 정보 및 프레임 저장 위치 관리

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
MODEL = os.getenv("QWEN_MODEL")

FRAME_DIR = "videos/frames"
RESULT_DIR = "results"