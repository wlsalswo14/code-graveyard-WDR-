import cv2
import numpy as np
import os
import random
import yaml
import shutil
import subprocess
import glob
from tqdm import tqdm

# --- 1. 전처리 클래스 (기존 로직 포함) ---
class GlareFusionPreprocessor:
    def __init__(self, raw_path, save_path, target_size=(640, 416)):
        self.raw_path = raw_path
        self.save_path = save_path
        self.target_size = target_size
        self.SATURATION_TRIGGER_RATIO = 0.05
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def apply_radial_glare(self, img, L=2, B_max=150):
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        diag = np.sqrt(w**2 + h**2)
        r = diag * 0.5
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = np.clip(1 - (dist / (r * L)), 0, 1)
        glare = mask * (B_max * L)
        img_glared = img.astype(np.float32) + cv2.merge([glare]*3)
        return np.clip(img_glared, 0, 255).astype(np.uint8)

    def pyramid_fusion(self, img_hs, img_binary, levels=3):
        gray_hs = cv2.cvtColor(img_hs, cv2.COLOR_BGR2GRAY)
        weight = np.clip((gray_hs.astype(np.float32) - 170) / 85, 0, 1)
        weight = cv2.GaussianBlur(weight, (11, 11), 0)
        weight_stack = cv2.merge([weight]*3)

        img1, img2 = img_hs.astype(np.float32), cv2.merge([img_binary]*3).astype(np.float32)
        
        gp_img1, gp_img2, gp_weight = [img1], [img2], [weight_stack]
        for _ in range(levels):
            img1, img2, weight_stack = cv2.pyrDown(img1), cv2.pyrDown(img2), cv2.pyrDown(weight_stack)
            gp_img1.append(img1); gp_img2.append(img2); gp_weight.append(weight_stack)

        lp_img1, lp_img2 = [gp_img1[levels]], [gp_img2[levels]]
        for i in range(levels, 0, -1):
            sz = (gp_img1[i-1].shape[1], gp_img1[i-1].shape[0])
            lp_img1.append(cv2.subtract(gp_img1[i-1], cv2.pyrUp(gp_img1[i], dstsize=sz)))
            lp_img2.append(cv2.subtract(gp_img2[i-1], cv2.pyrUp(gp_img2[i], dstsize=sz)))

        fused_lp = []
        for i in range(levels + 1):
            w = gp_weight[levels - i]
            fused_lp.append(lp_img2[i] * w + lp_img1[i] * (1.0 - w))

        fused_img = fused_lp[0]
        for i in range(1, levels + 1):
            sz = (fused_lp[i].shape[1], fused_lp[i].shape[0])
            fused_img = cv2.add(cv2.pyrUp(fused_img, dstsize=sz), fused_lp[i])
        return np.clip(fused_img, 0, 255).astype(np.uint8)

    def run(self):
        print(">> 이미지 융합 전처리를 시작합니다...")
        image_files = [f for f in os.listdir(self.raw_path) if f.endswith(('.png', '.jpg'))]
        for f in tqdm(image_files):
            img = cv2.imread(os.path.join(self.raw_path, f))
            if img is None: continue
            img = cv2.resize(img, self.target_size)
            
            img_ls = np.clip(img.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)
            
            # L값을 1.0 ~ 2.0 사이에서 랜덤하게 선택
            l_val = random.uniform(1.0, 2.0)
            img_hs = self.apply_radial_glare(img_ls, L=l_val)
            
            gray_hs = cv2.cvtColor(img_hs, cv2.COLOR_BGR2GRAY)
            sat_ratio = np.sum(gray_hs > 245) / (gray_hs.shape[0] * gray_hs.shape[1])
            
            # L값이 1.3 초과이고, 눈부심 비율이 임계치 이상일 때만 피라미드 융합 수행
            if l_val > 1.3 and sat_ratio > self.SATURATION_TRIGGER_RATIO:
                gray_ls = cv2.cvtColor(img_ls, cv2.COLOR_BGR2GRAY)
                thresh = 160 + (np.mean(gray_ls)/255)*70
                _, binary = cv2.threshold(gray_ls, thresh, 255, cv2.THRESH_BINARY)
                
                # 1. 피라미드 융합 (자연스러운 합성)
                fused = self.pyramid_fusion(img_hs, binary)
                
                # 2. 윤곽선 직접 오버레이 (구조 정보 강조)
                # 이진 이미지에서 윤곽선 추출
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # 윤곽선을 그릴 캔버스 생성 (검정 바탕)
                edge_img = np.zeros_like(fused)
                cv2.drawContours(edge_img, contours, -1, (255, 255, 255), 1)
                
                # 윤곽선이 있는 부분만 가중치를 두어 합성 (알파 블렌딩)
                alpha = 0.2  # 윤곽선 강조 강도 (0.2 = 20%)
                mask = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY) > 0
                output = fused.copy()
                output[mask] = cv2.addWeighted(fused, 1 - alpha, edge_img, alpha, 0)[mask]
            else:
                # L이 1.0~1.3 사이이거나 눈부심이 적은 경우는 단순 빛번짐 이미지 사용
                output = img_hs
            cv2.imwrite(os.path.join(self.save_path, f), output)

# --- 2. YOLO 학습 및 가중치 관리 함수 ---
def train_yolov7_tiny():
    # 학습 설정 값
    BATCH_SIZE = 16
    LR0 = 0.001  # 파인튜닝을 위해 학습률을 0.001로 하향 조정
    EPOCHS = 30
    IMG_HEIGHT = 416  # 600*400 비율(1.5)을 유지하기 위해 32의 배수인 416으로 설정
    IMG_WIDTH = 640   # 32의 배수인 640으로 설정
    MODEL_CONFIG = "cfg/training/yolov7-tiny.yaml"
    PRETRAINED_WEIGHTS = "yolov7-tiny.pt"
    
    # [설정] Top 3 가중치 저장을 위한 폴더 생성
    top_weights_dir = "runs/train/top_3_weights"
    if not os.path.exists(top_weights_dir):
        os.makedirs(top_weights_dir)

    # 하이퍼파라미터 파일 생성 (LR0 적용)
    hyp_src = "data/hyp.scratch.tiny.yaml"
    hyp_dst = "data/hyp.custom.yaml"
    hyp_file = hyp_src
    if os.path.exists(hyp_src):
        import re
        with open(hyp_src, "r") as f:
            hyp_content = f.read()
        hyp_content = re.sub(r'lr0:\s*[\d\.]+', f'lr0: {LR0}', hyp_content)
        with open(hyp_dst, "w") as f:
            f.write(hyp_content)
        hyp_file = hyp_dst
        print(f">> 커스텀 하이퍼파라미터 파일 생성 완료 (LR={LR0})")

    print(f">> YOLOv7-tiny 학습 시작 (Batch={BATCH_SIZE}, LR={LR0}, Size={IMG_WIDTH}x{IMG_HEIGHT}, Epochs={EPOCHS})")
    
    # YOLOv7 학습 명령어 실행 (subprocess 활용)
    cmd = [
        "python", "train.py",
        "--batch-size", str(BATCH_SIZE),
        "--data", "data/custom_data.yaml",
        "--img", str(IMG_HEIGHT), str(IMG_WIDTH),
        "--cfg", MODEL_CONFIG,
        "--weights", PRETRAINED_WEIGHTS,
        "--name", "yolov7_tiny_fused",
        "--hyp", hyp_file,
        "--epochs", str(EPOCHS),
        "--device", "0",
        "--save-period", "1"
    ]
    
    # 실제 학습 실행
    subprocess.run(cmd)

    # --- 학습 종료 후 Top 3 가중치 선별 로직 ---
    print(">> 학습 완료. 상위 성능 가중치를 정리합니다.")
    
    # 1. 최신 실행 폴더 찾기
    dirs = glob.glob("runs/train/yolov7_tiny_fused*")
    if not dirs:
        print(">> 학습 결과 폴더를 찾을 수 없습니다.")
        return
    latest_dir = max(dirs, key=os.path.getmtime)
    print(f">> 최신 결과 폴더: {latest_dir}")
    
    results_file = os.path.join(latest_dir, "results.txt")
    if not os.path.exists(results_file):
        print(">> results.txt 파일이 존재하지 않습니다.")
        return
        
    # 2. results.txt 분석하여 베스트 에폭(mAP@0.5 기준) 찾기
    all_results = []
    
    with open(results_file, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 12:
            continue
        try:
            epoch = int(parts[0])
            # YOLOv7 results.txt: index 10 is typically mAP@0.5
            map50 = float(parts[10])
            all_results.append((epoch, map50))
        except (ValueError, IndexError):
            continue
            
    # mAP@0.5 기준으로 내림차순 정렬 후 상위 3개 선택
    all_results.sort(key=lambda x: x[1], reverse=True)
    top_3 = all_results[:3]
    best_epochs = [x[0] for x in top_3]
            
    print(f">> 선택된 베스트 에폭 (Top 3): {best_epochs}")
    
    # 3. 가중치 파일 복사
    weights_dir = os.path.join(latest_dir, "weights")
    
    for i, ep in enumerate(best_epochs):
        # 파일 패턴 매칭 (YOLOv7은 epoch_N.pt 또는 epoch_00N.pt 등으로 저장할 수 있음)
        matches = glob.glob(os.path.join(weights_dir, f"epoch*{ep}.pt"))
        if not matches:
            # 직접 찾아보기
            matches = [f for f in os.listdir(weights_dir) if f"_{ep}.pt" in f or f"_{ep:03d}.pt" in f]
            matches = [os.path.join(weights_dir, f) for f in matches]
            
        if matches:
            src = matches[0]
            dst = os.path.join(top_weights_dir, f"best_top_{i+1}.pt")
            shutil.copy(src, dst)
            print(f">> {src} -> {dst} 복사 완료")
        else:
            print(f">> 에폭 {ep}에 대한 가중치 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    # 1. 이미지 전처리 실행
    preprocessor = GlareFusionPreprocessor(
        raw_path="data/pascalraw/images/JPEGImages/",
        save_path="data/train_fused_img/"
    )
    preprocessor.run()
    
    # 2. 학습 시작
    train_yolov7_tiny()