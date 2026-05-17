import os
import cv2
import numpy as np
import sys
import random

# src 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.augmentation import (
    simulate_high_glare, 
    generate_binary_from_saturated, 
    compute_threshold_prm, 
    calibrate_prm_trendline
)
from src import config

def run_sample_test(input_dir, output_dir, num_samples=5):
    """
    학습 데이터셋 중 일부를 무작위로 뽑아 
    실시간 고조도 시뮬레이션 및 이진화가 잘 되는지 시각적으로 확인합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"No images found in {input_dir}")
        return

    # 1. PRM 캘리브레이션 (샘플 100장 기준)
    sample_for_calib = random.sample(images, min(100, len(images)))
    calib_images = []
    for img_name in sample_for_calib:
        img = cv2.imread(os.path.join(input_dir, img_name))
        if img is not None:
            calib_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if calib_images:
        calibrate_prm_trendline(calib_images, target_ratio=config.TARGET_SATURATION_RATIO)
        print("PRM Calibration Completed for Sample Test.")

    # 2. 무작위 샘플 추출 및 결과 저장
    test_samples = random.sample(images, min(num_samples, len(images)))
    
    for idx, img_name in enumerate(test_samples):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 랜덤 L 값 적용 (0.0 ~ 2.0)
        L = np.random.uniform(0.0, 2.0)
        
        # 고조도 시뮬레이션
        glare_img, raw_irradiance = simulate_high_glare(image, L=L)
        
        # 임계값 동적 산출
        dyn_thresh = compute_threshold_prm(glare_img)
            
        # 이진 이미지 생성
        binary_img = generate_binary_from_saturated(raw_irradiance, threshold=dyn_thresh)
        
        # 결과 결합 (Original | Glare | Binary)
        h, w = image.shape[:2]
        # 시각화를 위해 RGB로 변환
        binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        
        combined = np.hstack([image, glare_img, binary_rgb])
        combined_resized = cv2.resize(combined, (config.IMAGE_SIZE[0] * 3, config.IMAGE_SIZE[1]))
        
        save_path = os.path.join(output_dir, f"sample_{idx}_L{L:.2f}_{img_name}")
        cv2.imwrite(save_path, cv2.cvtColor(combined_resized, cv2.COLOR_RGB2BGR))
        print(f"Saved sample result: {save_path}")

if __name__ == "__main__":
    print("Sample Test script started...")
    BASE_DIR = config.DATA_DIR
    INPUT_DIR = os.path.join(BASE_DIR, "images")
    OUTPUT_DIR = os.path.join(BASE_DIR, "debug_samples")

    # run_sample_test(INPUT_DIR, OUTPUT_DIR)
    print("Sample Test script completed.")
