import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.datasets import letterbox

def process_wdr_fusion(img):
    """
    원본 이미지를 기반으로 포화도를 검사하고, 
    조건 만족 시 동적 임계값 판단 후 이진화/엣지 융합을 수행합니다.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 원본 이미지의 포화도(Saturation) 계산
    sat_pixels = np.sum(gray > 245)
    sat_ratio = sat_pixels / (img.shape[0] * img.shape[1])
    SATURATION_TRIGGER_RATIO = 0.05
    
    if sat_ratio > SATURATION_TRIGGER_RATIO:
        # 2. 원본 이미지 평균 밝기 기반 동적 임계값 산출
        avg_val = np.mean(gray)
        thresh_val = 160 + (avg_val / 255) * 70
        
        # 3. 이진 이미지 생성 및 엣지 추출
        _, wdr_binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(wdr_binary, 80, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # 4. 포화 영역 마스크 생성 및 엣지와 교집합(AND) 연산
        sat_mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)[1]
        localized_edges = cv2.bitwise_and(edges_dilated, sat_mask)
        
        # 5. 원본 이미지와 융합 (포화된 엣지 영역에 검은색 윤곽선 주입)
        fused = img.copy()
        fused[localized_edges == 255] = [0, 0, 0] 
        return fused, True, sat_ratio
    else:
        # 포화도가 낮으면 원본 이미지 그대로 반환
        return img, False, sat_ratio

def run_wdr_inference(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 로드 (경로 확인 필요)
    model = attempt_load('yolov7.pt', map_location=device)
    
    # 원본 이미지 로드
    img = cv2.imread(img_path)
    if img is None: 
        print("이미지를 불러올 수 없습니다.")
        return
        
    # 필요에 따라 리사이즈 (YOLO 내부 letterbox가 있으나, 전처리 로직 통일을 위해 유지)
    img = cv2.resize(img, (600, 400))
    
    # --- 핵심 로직: 원본 이미지를 그대로 넘겨 융합 ---
    input_fused, is_fused, s_ratio = process_wdr_fusion(img)
    
    # YOLO 추론을 위한 전처리
    img_yolo = letterbox(input_fused, 640, stride=32)[0]
    img_yolo = img_yolo[:, :, ::-1].transpose(2, 0, 1)
    img_yolo = np.ascontiguousarray(img_yolo)
    img_yolo = torch.from_numpy(img_yolo).to(device).float() / 255.0
    if img_yolo.ndimension() == 3: 
        img_yolo = img_yolo.unsqueeze(0)

    # 추론 실행
    with torch.no_grad():
        pred = model(img_yolo)[0]
    pred = non_max_suppression(pred, 0.25, 0.45)

    # 시각화 텍스트
    mode_text = "FUSED MODE" if is_fused else "NORMAL MODE"
    info_text = f"WDR {mode_text} (Sat: {s_ratio:.1%})"
    cv2.putText(input_fused, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"[Inference Result] Fusion Applied: {is_fused}, Saturation: {s_ratio:.1%}")
    cv2.imshow('WDR Inference on Original Image', input_fused)
    cv2.waitKey(0)

if __name__ == "__main__":
    test_path = "data/pascalraw/JPEGImages/2014_000001.png"
    run_wdr_inference(test_path)