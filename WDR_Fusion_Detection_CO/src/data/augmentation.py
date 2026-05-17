import cv2
import numpy as np
import random

def simulate_high_glare(image, center=None, r=1200, B_max=150, L=1.0):
    """
    빛 번짐 적용 함수. 일반 이미지(클리핑됨)와 물리적 광량 데이터(클리핑 안 됨)를 모두 반환합니다.
    """
    h, w = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if center is None:
        center = (w // 2, h // 2)
        
    max_pixel_pos = np.unravel_index(gray.argmax(), gray.shape)[::-1]
    
    Y, X = np.ogrid[:h, :w]
    dist_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    dist_max = np.sqrt((X - max_pixel_pos[0])**2 + (Y - max_pixel_pos[1])**2)
    
    mask_center = np.maximum(0, 1.0 - dist_center / (r * L))
    mask_max = np.maximum(0, 1.0 - dist_max / (r * L))
    mask = np.maximum(mask_center, mask_max)
    
    # 1. 물리적 광량(Irradiance) 보존 (float32 유지, 255 클리핑 없음)
    raw_irradiance = image.astype(np.float32)
    raw_irradiance += mask[..., np.newaxis] * (B_max * L)
    
    # 2. 일반 카메라용 Saturated Image (uint8 클리핑 적용)
    glare_image = np.clip(raw_irradiance, 0, 255).astype(np.uint8)
    
    # 두 데이터를 모두 반환
    return glare_image, raw_irradiance

def simulate_high_glare_online(image, epoch=None, L_override=None, jitter_ratio=0.05):
    """
    수정된 Online Augmentation 전략 (Jittering):
    이미지 중앙에서 미세하게(±5%)만 흔들고, 세기와 크기(±10~20%)에 랜덤성을 부여하여
    객체 은폐를 유지하면서 픽셀 중복을 방지합니다.
    """
    h, w = image.shape[:2]

    # 평가/디버깅 용도로 glare를 완전히 끄는 케이스 지원
    if L_override is not None and float(L_override) <= 0.0:
        raw_irradiance = image.astype(np.float32)
        return image.copy(), raw_irradiance
    
    # 1. 완전 무작위가 아닌 중앙값 기반의 미세한 흔들림(Jittering)
    # 이미지 중앙에서 ±5% 이내로만 미세하게 중심점 이동 (객체를 계속 가리도록)
    jitter_x = random.uniform(-jitter_ratio, jitter_ratio) * w
    jitter_y = random.uniform(-jitter_ratio, jitter_ratio) * h
    center = (int(w / 2 + jitter_x), int(h / 2 + jitter_y))
    
    # 2. 빛의 크기(r)와 세기(B_max, L)만 랜덤성 부여
    r = random.uniform(1000, 1400)      # 논문 기본값 1200 주변
    B_max = random.uniform(130, 170)    # 논문 기본값 150 주변
    # Curriculum L: warm-up(초기)에는 난이도 완화
    # - warm-up:   L ∈ [0, GLARE_L_MAX_WARMUP]
    # - 이후:      L ∈ [0, GLARE_L_MAX_AFTER_WARMUP]
    if L_override is not None:
        L = float(L_override)
    elif epoch is not None:
        # Phase 4 Linear Curriculum: L starts at 0.5, increases by 0.1 per epoch, capped at 1.5
        L_base = 0.5 + (epoch * 0.1)
        L = random.uniform(max(0.5, L_base - 0.1), min(1.5, L_base + 0.1))
        L = min(L, 1.5)
    
    # 기존 시뮬레이션 로직 재사용
    return simulate_high_glare(image, center=center, r=r, B_max=B_max, L=L)

def generate_binary_from_saturated(raw_irradiance, threshold=240, saturation_limit=253):
    """
    WDR 이진화 함수. 클리핑되지 않은 원본 광량을 기반으로 LS 픽셀 정보를 계산합니다.
    
    OTR(포화) 영역과 비포화 영역에 각각 스케일에 맞는 threshold를 적용합니다.
    - OTR 영역: LS 픽셀값은 fill factor 비율(8.4/38.5 ≈ 0.218)로 스케일되므로
      threshold도 동일 비율로 스케일하여 적용합니다.
    - 비포화 영역: 원래 threshold를 그대로 적용합니다.
    """
    FILL_FACTOR_RATIO = 8.4 / 38.5  # ≈ 0.218

    # RGB 광량을 Gray(단일 채널) 광량으로 변환 
    # (cv2.cvtColor는 클리핑을 유발할 수 있으므로 가중치 공식으로 수동 변환)
    gray_irradiance = (0.299 * raw_irradiance[..., 0] + 
                       0.587 * raw_irradiance[..., 1] + 
                       0.114 * raw_irradiance[..., 2])
    
    # DOTR (Out of Range 판단): 빛의 양이 포화 기준을 넘었는지 확인
    d_otr = (gray_irradiance >= saturation_limit)
    
    # LS 픽셀 연산: **포화되기 전의 쌩 데이터(gray_irradiance)**에 Fill Factor 비율을 곱함
    # 이 연산 덕분에 하얗게 날아간 영역 내부의 윤곽선 디테일이 살아납니다.
    ls_gray = gray_irradiance * FILL_FACTOR_RATIO
    
    # HS 픽셀(일반 감도)은 255로 클리핑된 일반 픽셀값 사용
    hs_gray = np.clip(gray_irradiance, 0, 255)
    
    # OTR/비OTR 영역별로 스케일에 맞는 threshold를 각각 적용하여 이진화
    # LS 영역: threshold도 fill factor 비율로 스케일 (예: 240 * 0.218 ≈ 52.4)
    ls_threshold = threshold * FILL_FACTOR_RATIO
    
    binary = np.zeros_like(gray_irradiance, dtype=np.uint8)
    binary[d_otr & (ls_gray >= ls_threshold)] = 255       # 포화 영역: LS 스케일 threshold
    binary[~d_otr & (hs_gray >= threshold)] = 255          # 비포화 영역: 원래 threshold
    
    return binary

_prm_trendline_func = None

def calibrate_prm_trendline(images, target_ratio=0.05):
    """
    학습 데이터셋(이미지 리스트)을 순회하며 평균 픽셀 값에 따른 백색/흑색 픽셀 비율의
    상관관계를 분석하여 2차 다항식 형태의 추세선 함수(Trendline function)를 생성합니다.
    """
    global _prm_trendline_func
    means = []
    best_thresholds = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_val = gray.mean()
        
        # 최적 임계값 탐색 (target_ratio에 가장 가까운 비율을 내는 임계값)
        best_t = 128
        min_diff = float('inf')
        for t in range(50, 250, 5):
            ratio = (gray > t).mean()
            if abs(ratio - target_ratio) < min_diff:
                min_diff = abs(ratio - target_ratio)
                best_t = t
                
        means.append(mean_val)
        best_thresholds.append(best_t)
        
    if len(means) > 1:
        # 2차 다항식 추세선 피팅
        coeffs = np.polyfit(means, best_thresholds, 2)
        _prm_trendline_func = np.poly1d(coeffs)
    else:
        _prm_trendline_func = np.poly1d([0.0, -1.0, 255.0])

def compute_threshold_prm(image):
    """
    입력 이미지의 평균 픽셀 값을 사전에 도출된 추세선 함수(Trendline function)에 대입하여 
    최적의 임계값(Threshold)을 동적으로 반환
    """
    global _prm_trendline_func
    if _prm_trendline_func is None:
        # 캘리브레이션 전 기본 다항식
        # 포화 이미지(평균 밝기 ~150-200)에서 합리적인 threshold를 반환하도록 설계:
        #   f(100) ≈ 200, f(150) ≈ 170, f(200) ≈ 150
        _prm_trendline_func = np.poly1d([0.002, -0.9, 230])
        
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_val = gray.mean()
    
    threshold = _prm_trendline_func(mean_val)
    return np.clip(threshold, 50, 250).astype(np.uint8)

def compute_threshold_otsu(image, update_every_n=10, frame_idx=0, prev_threshold=128):
    if frame_idx % update_every_n == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold
    return prev_threshold
