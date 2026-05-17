import os
import sys
import torch
import argparse
import random
import cv2
import importlib.util

# PyTorch 2.6+ 호환성 문제 해결
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

from torch.utils.data import DataLoader, random_split

# 프로젝트 루트·yolov7 서브모듈을 import 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov7")))

from src.data.augmentation import calibrate_prm_trendline, simulate_high_glare_online
_fusion_training_path = os.path.join(os.path.dirname(__file__), "02_fusion_training.py")
_spec = importlib.util.spec_from_file_location("fusion_training", _fusion_training_path)
_fusion_training = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_fusion_training)

train_fusion_model = _fusion_training.train_fusion_model
finetune_yolo_model = _fusion_training.finetune_yolo_model
from src.data.dataset import WDRFusionDataset, wdr_collate_fn
from src import config

def run_training(images_dir, labels_dir, epochs_stage1, epochs_stage2):
    print("==================================================")
    print("🚀 WDR Fusion Pipeline Full Training (Online Augment)")
    print("==================================================")
    
    # 1. PRM 캘리브레이션 (Phase 4 추가 요구사항)
    print("\n[Step 1] PRM 추세선 캘리브레이션 시작...")
    original_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(original_images) == 0:
        print(f"[오류] 이미지 디렉토리가 비어있습니다: {images_dir}")
        return
        
    # 무작위로 300장의 시뮬레이션 샘플 생성 (Phase 4 강화)
    target_sample_count = 300
    samples = []
    
    if len(original_images) > 0:
        # 원본 이미지에서 무작위 샘플링 (필요 시 중복 허용하여 100장 채움)
        sample_files = random.choices(original_images, k=target_sample_count)
        
        for f in sample_files:
            img_path = os.path.join(images_dir, f)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # [Phase 4] 다양한 조도(L=0.5~2.0) 대응을 위해 추세선 데이터 다변화
                glare_img, _ = simulate_high_glare_online(img_rgb, L_override=random.uniform(0.5, 2.0))
                samples.append(glare_img)
    
    if samples:
        calibrate_prm_trendline(samples, target_ratio=config.TARGET_SATURATION_RATIO)
        print(f"✅ {len(samples)}장의 시뮬레이션 샘플로 PRM 캘리브레이션 완료.")
    else:
        print("⚠️ 샘플 이미지를 생성하지 못했습니다. 기본 설정을 사용합니다.")

    # 2. 데이터로더 생성 (Train / Val 분리)
    print("\n[Step 2] 데이터셋 로드 및 Train/Val 분리...")
    full_dataset = WDRFusionDataset(
        original_dir=images_dir,
        labels_dir=labels_dir,
        img_size=config.IMAGE_SIZE
    )
    
    total_size = len(full_dataset)
    if total_size == 0:
        print("[오류] 데이터셋이 비어있습니다. data/images 와 data/labels(또는 --images_dir/--labels_dir)를 확인하세요.")
        return
        
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = config.BATCH_SIZE
    num_workers = getattr(config, 'NUM_WORKERS', 4)
    pin_memory = getattr(config, 'PIN_MEMORY', True)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=wdr_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=wdr_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"[Step 2] 분리 완료! 총 데이터: {total_size}장 (학습용: {train_size}장 / 검증용: {val_size}장)")
    print(f"         배치 사이즈: {batch_size}, Workers: {num_workers}, Pin Memory: {pin_memory}")

    # 3. YOLOv7 사전학습 모델 로드
    print("\n[Step 3] YOLOv7 사전학습 모델 로드 (COCO Pretrained)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 기기: {device}")
    
    try:
        from models.experimental import attempt_load
        
        pretrained_path = config.find_yolov7_pretrained("yolov7-tiny.pt")

        if pretrained_path is None:
            print("[오류] 사전학습 가중치(yolov7-tiny.pt)를 찾을 수 없습니다.")
            print("      프로젝트 루트·weights/에 배치하거나 YOLO_PRETRAINED_PATH 환경변수를 설정하세요.")
            return
        
        yolo_model = attempt_load(pretrained_path, map_location=device)
        yolo_model = yolo_model.to(device)
        print(f"[Step 3] YOLOv7 사전학습 모델 로드 완료: {pretrained_path}")
    except ImportError:
        print("[오류] YOLOv7 모듈을 찾을 수 없습니다.")
        return

    # 4. Fusion Model 학습 (Stage 1)
    print(f"\n[Step 4] Stage 1: Fusion Net 학습 ({epochs_stage1} Epochs)...")
    fusion_model = train_fusion_model(train_dataloader, val_dataloader, yolo_model, epochs=epochs_stage1)
    print("[Step 4] Stage 1 완료!")

    # 5. YOLO Finetuning (Stage 2)
    print(f"\n[Step 5] Stage 2: YOLOv7 파인튜닝 ({epochs_stage2} Epochs)...")
    from src.models.yolo_wrapper import YoloWrapper
    yolo_wrapper = YoloWrapper(yolo_model).to(device)
    
    finetuned_yolo = finetune_yolo_model(train_dataloader, fusion_model, yolo_wrapper, val_dataloader=val_dataloader, epochs=epochs_stage2)
    print("[Step 5] Stage 2 완료!")
    
    print("\n==================================================")
    print("🎉 모든 학습이 성공적으로 종료되었습니다!")
    print(" output/checkpoints/ 폴더에 베스트 모델들이 저장되었습니다.")
    print("==================================================")

def _resolve_data_paths(args):
    """--data_dir(=--base_dir) 또는 개별 디렉터리로 images/labels 경로 확정."""
    if args.images_dir and args.labels_dir:
        return os.path.abspath(args.images_dir), os.path.abspath(args.labels_dir)
    if args.data_dir:
        root = os.path.abspath(args.data_dir)
        return os.path.join(root, "images"), os.path.join(root, "labels")
    return config.DATA_IMAGES_DIR, config.DATA_LABELS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WDR Fusion 학습 (로컬/클라우드 공통). 기본 데이터: 프로젝트의 data/images, data/labels"
    )
    parser.add_argument('--epochs_stage1', type=int, default=getattr(config, 'STAGE1_EPOCHS', getattr(config, 'EPOCHS', 50)), help='Stage 1 (FusionNet) Epochs')
    parser.add_argument('--epochs_stage2', type=int, default=getattr(config, 'STAGE2_EPOCHS', getattr(config, 'EPOCHS', 50)), help='Stage 2 (YOLOv7) Epochs')
    parser.add_argument(
        '--data_dir',
        '--base_dir',
        dest='data_dir',
        type=str,
        default=None,
        help='데이터 루트(images/, labels/ 포함). 미지정 시 <프로젝트>/data',
    )
    parser.add_argument('--images_dir', type=str, default=None, help='원본 이미지 폴더 (지정 시 --labels_dir 필수)')
    parser.add_argument('--labels_dir', type=str, default=None, help='YOLO 라벨(.txt) 폴더')
    args = parser.parse_args()

    if (args.images_dir or args.labels_dir) and not (args.images_dir and args.labels_dir):
        print("[오류] --images_dir 와 --labels_dir 는 함께 지정해야 합니다.")
        raise SystemExit(1)

    IMAGES_DIR, LABELS_DIR = _resolve_data_paths(args)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    run_training(IMAGES_DIR, LABELS_DIR, args.epochs_stage1, args.epochs_stage2)
