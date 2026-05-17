import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import multiprocessing
from src import config
from src.data.augmentation import (
    simulate_high_glare, 
    simulate_high_glare_online,
    generate_binary_from_saturated, 
    compute_threshold_prm
)

# Python <3.10 호환용 typing
from typing import Optional

class WDRFusionDataset(Dataset):
    def __init__(
        self,
        original_dir,
        labels_dir=None,
        transform=None,
        img_size=config.IMAGE_SIZE,
        glare_mode: str = "online",
        fixed_L: Optional[float] = None,
    ):
        """
        Args:
            original_dir (str): 깨끗한 원본 이미지가 있는 디렉토리
            labels_dir (str, optional): YOLO 타겟 라벨(txt)이 있는 디렉토리
            transform (callable, optional): 이미지 및 텐서 변환 적용
            img_size (tuple, optional): (width, height) 이미지 리사이즈 크기. 기본값 (416, 416)
            glare_mode (str): "online"(train), "fixed"(eval), "none"(L=0 평가)
            fixed_L (float | None): glare_mode=="fixed"일 때 사용할 L
        """
        self.original_dir = original_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_size = img_size
        self.glare_mode = glare_mode
        self.fixed_L = fixed_L
        # 📌 멀티프로세싱(NUM_WORKERS > 0) 환경에서 에포크 동기화를 위해 공유 메모리 사용
        self._current_epoch = multiprocessing.Value('i', 0)
        
        self.image_files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    @property
    def current_epoch(self):
        return self._current_epoch.value

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch.value = int(value)

    def __len__(self):
        return len(self.image_files)

    def load_image(self, path):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 리사이즈는 augmentation 전에 수행하여 속도 최적화 가능하지만, 
            # 원본 디테일 보존을 위해 augmentation 후 리사이즈 권장.
            # 여기서는 일관성을 위해 load 단계에서 기본 리사이즈 수행.
            if img.shape[:2] != self.img_size[::-1]:
                img = cv2.resize(img, self.img_size)
        return img

    def load_targets(self, img_name):
        if not self.labels_dir:
            return torch.zeros((0, 5))
            
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # 📌 COCO 사전학습 모델 호환성을 위해 ID 매핑 적용
                        if hasattr(config, 'PASCAL_TO_COCO_MAP'):
                            class_id = float(config.PASCAL_TO_COCO_MAP.get(int(class_id), class_id))
                            
                        targets.append([class_id, cx, cy, w, h])
        
        return torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 5))

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # 1. 원본 깨끗한 이미지 로드
        original_path = os.path.join(self.original_dir, img_name)
        original_img = self.load_image(original_path)
        
        if original_img is None:
            raise FileNotFoundError(f"Could not load image: {original_path}")

        # 2. 고조도 시뮬레이션 (Train/Eval 모드 분리)
        if self.glare_mode == "none":
            saturated_img = original_img
            raw_irradiance = original_img.astype(np.float32)
        elif self.glare_mode == "fixed":
            saturated_img, raw_irradiance = simulate_high_glare_online(
                original_img,
                epoch=None,
                L_override=self.fixed_L,
            )
        else:
            # "online": epoch 기반 curriculum + jitter
            saturated_img, raw_irradiance = simulate_high_glare_online(
                original_img,
                epoch=self.current_epoch,
                L_override=None,
            )
        
        # 3. 이진화 이미지(binary_img) 즉석 생성
        dyn_thresh = compute_threshold_prm(saturated_img)
        binary_img = generate_binary_from_saturated(raw_irradiance, threshold=dyn_thresh)
        
        # binary_img는 (H, W) 이므로 (H, W, 1)로 변경
        binary_img = np.expand_dims(binary_img, axis=-1)

        # 4. YOLO용 타겟 데이터
        targets = self.load_targets(img_name)
        
        # 텐서 변환 및 정규화
        if self.transform:
            original_tensor = self.transform(original_img)
            saturated_tensor = self.transform(saturated_img)
        else:
            # Default Transform (HWC -> CHW, /255.0)
            original_tensor = torch.from_numpy(original_img).permute(2, 0, 1).float() / 255.0
            saturated_tensor = torch.from_numpy(saturated_img).permute(2, 0, 1).float() / 255.0
            
        binary_tensor = torch.from_numpy(binary_img).permute(2, 0, 1).float() / 255.0
            
        return original_tensor, saturated_tensor, binary_tensor, targets

def wdr_collate_fn(batch):
    originals, saturateds, binaries, targets = zip(*batch)
    
    originals = torch.stack(originals, 0)
    saturateds = torch.stack(saturateds, 0)
    binaries = torch.stack(binaries, 0)
    
    batched_targets = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            batch_idx_tensor = torch.full((t.shape[0], 1), i, dtype=torch.float32)
            t_with_batch = torch.cat([batch_idx_tensor, t], dim=1)
            batched_targets.append(t_with_batch)
            
    if batched_targets:
        batched_targets = torch.cat(batched_targets, 0)
    else:
        batched_targets = torch.zeros((0, 6))
        
    return originals, saturateds, binaries, batched_targets
