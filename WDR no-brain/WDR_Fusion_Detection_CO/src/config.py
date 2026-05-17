import os
from typing import Optional

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
# 사용자가 `data/` 아래에 두는 기본 레이아웃: images/, labels/ (YOLO txt)
DATA_IMAGES_DIR = os.path.join(DATA_DIR, "images")
DATA_LABELS_DIR = os.path.join(DATA_DIR, "labels")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")


def find_yolov7_pretrained(filename: str = "yolov7-tiny.pt") -> Optional[str]:
    """
    COCO 사전학습 가중치 경로를 탐색합니다.
    우선순위: 환경변수 YOLO_PRETRAINED_PATH → 프로젝트 루트/weights/ 등.
    """
    env_path = os.environ.get("YOLO_PRETRAINED_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    search_roots = [BASE_DIR, os.getcwd()]
    subdirs = ("", "weights", "checkpoints")
    for root in search_roots:
        for sub in subdirs:
            candidate = os.path.join(root, sub, filename) if sub else os.path.join(root, filename)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return None

# Hyperparameters (Paper specific: V-A)
ALPHA = 1.0
BETA = 1.2
GAMMA = 1.5

# Training
BATCH_SIZE = 8  # T4 16GB 기준 OOM 방지를 위해 8 유지
IMAGE_SIZE = (416, 416)
LEARNING_RATE = 0.001
LR_GAMMA = 0.97  # 제한된 에포크(50) 내에서 충분한 수렴을 위해 감쇠율 추가 상향 (0.95 -> 0.97)

# Stage schedule (저자원 안정형)
# Stage1 합계 50 = warm-up 10 + gamma ramp 5 + detection-aware(full) 35
WARMUP_EPOCHS = 0
STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 8

# Stage1 안정화: warm-up 후 det loss 가중치 ramp
# - 0이면 step(즉시 GAMMA 적용)
# - 5~10 권장 (저자원 불안정 시)
GAMMA_RAMP_EPOCHS = 4

# Stage2 (YOLO finetune)
YOLO_LEARNING_RATE = 5e-4

# Glare augmentation (train-time)
# 정보 소실 과다 방지 및 학습 안정성을 위해 L의 상한을 1.5로 조정
GLARE_L_MAX_WARMUP = 1.0
GLARE_L_MAX_AFTER_WARMUP = 1.5

# Stability toggles
USE_AMP = True
ACCUMULATION_STEPS = 1  # BATCH_SIZE 8 기준 매 배치마다 업데이트

# Notebook 기본값 (기존 EPOCHS 사용 코드 호환)
EPOCHS = STAGE1_EPOCHS

# Data Loading Optimization
NUM_WORKERS = 4  # 속도 향상을 위해 4로 상향
PIN_MEMORY = True
USE_COMPILE = False  # baseline 안정성을 위해 기본 비활성화

# Threshold logic parameters
TARGET_SATURATION_RATIO = 0.05
OTSU_UPDATE_INTERVAL = 10

# Custom Dataset (차, 사람, 자전거) -> COCO (80) ID Mapping
# 사용자 데이터셋 ID: 0:차, 1:사람, 2:자전거
# COCO 사전학습 모델 ID: 0:person, 1:bicycle, 2:car
PASCAL_TO_COCO_MAP = {
    0: 2,   # 차 -> car
    1: 0,   # 사람 -> person
    2: 1    # 자전거 -> bicycle
}
