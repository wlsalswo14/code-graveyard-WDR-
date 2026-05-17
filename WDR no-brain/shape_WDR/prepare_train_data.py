import cv2
import numpy as np
import os
import random
from tqdm import tqdm

def apply_radial_glare_L2(img, L=2, B_max=150):
    h, w = img.shape[:2]
    cx = (w // 2) + random.randint(-int(w * 0.05), int(w * 0.05))
    cy = (h // 2) + random.randint(-int(h * 0.05), int(h * 0.05))
    diag = np.sqrt(w**2 + h**2)
    r = diag * 0.5 
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = np.clip(1 - (dist / (r * L)), 0, 1)
    glare = mask * (B_max * L)
    img_glared = img.astype(np.float32) + cv2.merge([glare]*3)
    return np.clip(img_glared, 0, 255).astype(np.uint8)

def process_and_save():
    raw_path = "data/pascalraw/images/JPEGImages/"
    save_path = "data/pascalraw/train_fused_high_contrast/" 
    target_size = (600, 400)
    SATURATION_TRIGGER_RATIO = 0.05 
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_files = [f for f in os.listdir(raw_path) if f.endswith('.png')]
    print(f"고대비(Black-Edge) 국소 합성 데이터를 생성합니다 (L=2)...")

    for f in tqdm(image_files):
        img = cv2.imread(os.path.join(raw_path, f))
        if img is None: continue
        img = cv2.resize(img, target_size)

        # --- 이 부분을 수정합니다 ---
        # 1.0 ~ 1.5 사이의 랜덤한 float 값 생성
        random_L = random.uniform(1.0, 1.5) 
        
        # 1. HS 및 LS 생성 (랜덤 생성된 random_L 적용)
        img_ls = np.clip(img.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
        img_hs_glared = apply_radial_glare_L2(img_ls, L=random_L)

        # ---------------------------------------------------------
        # [수정] sat_ratio 및 gray_hs 계산 로직 추가
        gray_hs = cv2.cvtColor(img_hs_glared, cv2.COLOR_BGR2GRAY)
        sat_pixels = np.sum(gray_hs > 245)
        sat_ratio = sat_pixels / (gray_hs.shape[0] * gray_hs.shape[1])
        # ---------------------------------------------------------

        if sat_ratio > SATURATION_TRIGGER_RATIO:
            # [High-Contrast Localized Fusion]
            gray_ls = cv2.cvtColor(img_ls, cv2.COLOR_BGR2GRAY)
            avg_val = np.mean(gray_ls)
            thresh_val = 160 + (avg_val / 255) * 70 
            _, wdr_binary = cv2.threshold(gray_ls, thresh_val, 255, cv2.THRESH_BINARY)
            
            # 엣지 추출 및 가독성을 위해 조금 더 두껍게 확장
            edges = cv2.Canny(wdr_binary, 80, 150)
            edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

            # 포화 마스크 (이제 에러가 나지 않습니다)
            saturation_mask = cv2.threshold(gray_hs, 235, 255, cv2.THRESH_BINARY)[1]
            localized_edges = cv2.bitwise_and(edges_dilated, saturation_mask)

            # [고대비 적용] 포화된 밝은 영역(255)에 검은색(0) 윤곽선을 주입
            fused = img_hs_glared.copy()
            fused[localized_edges == 255] = [0, 0, 0] # 블랙 윤곽선으로 강한 대비 부여
            output_img = fused
        else:
            output_img = img_hs_glared

        cv2.imwrite(os.path.join(save_path, f), output_img)

if __name__ == "__main__":
    process_and_save()