import os
import sys
import argparse
import torch

# PyTorch 2.6+ 호환성 문제 해결 (weights_only 기본값 변경 방지)
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

from torch.utils.data import DataLoader

# 프로젝트 루트·yolov7 서브모듈을 import 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov7")))

import importlib.util

_fusion_training_path = os.path.join(os.path.dirname(__file__), "02_fusion_training.py")
_spec = importlib.util.spec_from_file_location("fusion_training", _fusion_training_path)
_fusion_training = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_fusion_training)

train_fusion_model = _fusion_training.train_fusion_model
finetune_yolo_model = _fusion_training.finetune_yolo_model
from src.data.dataset import WDRFusionDataset, wdr_collate_fn
from src import config


def _resolve_data_paths(args):
    if args.images_dir and args.labels_dir:
        return os.path.abspath(args.images_dir), os.path.abspath(args.labels_dir)
    if args.data_dir:
        root = os.path.abspath(args.data_dir)
        return os.path.join(root, "images"), os.path.join(root, "labels")
    return config.DATA_IMAGES_DIR, config.DATA_LABELS_DIR


def run_sanity_check(images_dir, labels_dir):
    print("=====================================")
    print("🚀 WDR Fusion Pipeline Sanity Check (Overfit Test)")
    print("=====================================")

    # 데이터로더 (온라인 glare·binary — 별도 전처리 디렉터리 불필요)
    print("\n[Step 1] 데이터로더 초기화...")
    dataset = WDRFusionDataset(
        original_dir=images_dir,
        labels_dir=labels_dir
    )

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=wdr_collate_fn)
    print(f"[Step 1] 데이터로더 준비 완료! (총 {len(dataset)}장)")

    # YOLOv7 COCO 사전학습 모델 로드
    print("\n[Step 2] YOLOv7 COCO 사전학습 모델 로드...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 기기: {device}")

    try:
        from models.experimental import attempt_load

        pretrained_path = config.find_yolov7_pretrained("yolov7-tiny.pt")

        if pretrained_path is None:
            print("[오류] COCO 사전학습 가중치 파일(yolov7-tiny.pt)을 찾을 수 없습니다.")
            print("       프로젝트 루트 또는 weights/에 두거나 YOLO_PRETRAINED_PATH 환경변수를 지정하세요.")
            print("       다운로드: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt")
            return

        yolo_model = attempt_load(pretrained_path, map_location=device)
        yolo_model = yolo_model.to(device)
        print(f"[Step 2] YOLOv7 COCO 사전학습 모델 로드 완료: {pretrained_path}")
    except ImportError:
        print("[오류] YOLOv7 모듈을 찾을 수 없습니다. yolov7 저장소가 올바르게 클론되었는지 확인하세요.")
        return

    print("\n[Step 3] Stage 1: Fusion Net 학습 (Gradient 확인)...")
    epochs_stage1 = 50
    fusion_model = train_fusion_model(dataloader, dataloader, yolo_model, epochs=epochs_stage1)
    print("[Step 3] Stage 1 완료!")

    print("\n[Step 4] Stage 2: YOLOv7 파인튜닝 (Gradient 확인)...")
    from src.models.yolo_wrapper import YoloWrapper
    yolo_wrapper = YoloWrapper(yolo_model).to(device)

    epochs_stage2 = 50
    finetune_yolo_model(dataloader, fusion_model, yolo_wrapper, val_dataloader=dataloader, epochs=epochs_stage2)
    print("[Step 4] Stage 2 완료!")

    print("\n=====================================")
    print("🎉 Sanity Check가 성공적으로 종료되었습니다!")
    print("파이프라인의 순전파, 오차 계산, 역전파가 정상 동작합니다.")
    print("=====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="소량 데이터 오버핏/동작 확인")
    parser.add_argument(
        '--data_dir',
        '--base_dir',
        dest='data_dir',
        type=str,
        default=None,
        help='데이터 루트(images/, labels/). 기본: <프로젝트>/data',
    )
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--labels_dir', type=str, default=None)
    args = parser.parse_args()

    if (args.images_dir and not args.labels_dir) or (args.labels_dir and not args.images_dir):
        print("[오류] --images_dir 와 --labels_dir 를 함께 지정해야 합니다.")
        raise SystemExit(1)

    IMAGES_DIR, LABELS_DIR = _resolve_data_paths(args)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    images = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if len(images) == 0:
        print(f"알림: '{IMAGES_DIR}' 폴더에 이미지가 없습니다.")
        print("테스트용 이미지를 넣은 뒤 다시 실행해 주세요.")
    else:
        for img in images:
            label_name = os.path.splitext(img)[0] + '.txt'
            label_path = os.path.join(LABELS_DIR, label_name)
            if not os.path.exists(label_path):
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.2 0.2\n")

        run_sanity_check(IMAGES_DIR, LABELS_DIR)
