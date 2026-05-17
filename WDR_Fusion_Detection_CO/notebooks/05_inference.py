"""
05_inference.py — WDR Fusion + YOLOv7 추론(Inference) 스크립트

사용법:
  1. 단일 이미지 추론:
     python 05_inference.py --image /path/to/image.jpg

  2. 폴더 내 전체 이미지 추론:
     python 05_inference.py --image_dir /path/to/images/

  3. 체크포인트 경로 지정 (기본: output/checkpoints/ 하위):
     python 05_inference.py --image /path/to/image.jpg \
         --fusion_ckpt /path/to/best_fusion_model.pth \
         --yolo_weights /path/to/yolov7.pt

  출력: output/inference_results/ 에 결과 이미지 저장
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import sys
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "yolov7"))

from src.models.fusion_net import FusionNet
from src import config
from src.data.augmentation import (
    simulate_high_glare,
    generate_binary_from_saturated,
    compute_threshold_prm,
)

# ─── 클래스 이름 정의 ──────────────────────────────────────────────────
# PascalRaw (Pascal VOC 기반) 20개 클래스
PASCAL_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO 80개 클래스 (필요시 사용)
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 현재 데이터셋에 맞는 클래스 선택 (COCO 모델인 yolov7-tiny.pt를 쓸 때)
CURRENT_CLASS_NAMES = COCO_NAMES

# 바운딩박스 컬러 팔레트 (클래스별 구분)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CURRENT_CLASS_NAMES), 3), dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# 1. 모델 로딩
# ═══════════════════════════════════════════════════════════════════════

def load_fusion_model(checkpoint_path, device):
    """학습된 FusionNet 가중치를 로드합니다."""
    model = FusionNet(out_channels=3).to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[✓] FusionNet 로드 완료: {checkpoint_path}")
    else:
        print(f"[!] FusionNet 체크포인트 없음: {checkpoint_path}")
        print("    학습되지 않은 랜덤 가중치로 추론합니다.")

    model.eval()
    return model


def load_yolo_model(weights_path, device, yolo_cfg=None):
    """
    YOLOv7 모델을 로드합니다.
    
    Stage 2에서 저장한 state_dict (.pth)와 사전학습된 전체 모델 (.pt) 모두 지원합니다.
    - yolo_cfg 지정 시: yaml로 모델 생성 후 state_dict 로드 (Stage 2 파인튜닝 가중치용)
    - yolo_cfg 미지정 시: attempt_load로 전체 모델 로드 (사전학습 yolov7.pt용)
    """
    if yolo_cfg and os.path.exists(yolo_cfg):
        # yaml 기반 모델 생성 + state_dict 로드 (Stage 2 저장 방식과 일치)
        from models.yolo import Model as YoloModel
        model = YoloModel(yolo_cfg, ch=3, nc=80).to(device)
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[✓] YOLOv7 로드 완료 (yaml + state_dict): {weights_path}")
        else:
            print(f"[!] YOLOv7 가중치 없음: {weights_path}")
            print("    학습되지 않은 랜덤 가중치로 추론합니다.")
    else:
        # 사전학습된 전체 모델 로드 (.pt)
        from models.experimental import attempt_load
        model = attempt_load(weights_path, map_location=device)
        print(f"[✓] YOLOv7 로드 완료 (pretrained): {weights_path}")

    model = model.to(device)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# 2. 이미지 전처리 (Saturated + Binary 생성)
# ═══════════════════════════════════════════════════════════════════════

def preprocess_image(image_path, img_size=416, glare_L=None):
    """
    원본 이미지에서 Saturated + Binary 텐서를 생성합니다.

    Returns:
        saturated_tensor: (1, 3, H, W) float32 [0, 1]
        binary_tensor:    (1, 1, H, W) float32 [0, 1]
        original_bgr:     원본 이미지 (시각화용, BGR)
        scale_info:       리사이즈 비율 정보 (bbox 역변환용)
    """
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {image_path}")

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_rgb.shape[:2]

    # 고조도 시뮬레이션 → Saturated + Raw Irradiance
    # glare_L이 0이면 glare 없이 평가 가능
    if glare_L is not None and float(glare_L) <= 0.0:
        saturated = original_rgb
        raw_irradiance = original_rgb.astype(np.float32)
    elif glare_L is not None:
        saturated, raw_irradiance = simulate_high_glare(original_rgb, L=float(glare_L))
    else:
        saturated, raw_irradiance = simulate_high_glare(original_rgb)

    # 동적 임계값 → Binary 생성
    threshold = compute_threshold_prm(saturated)
    binary = generate_binary_from_saturated(raw_irradiance, threshold=threshold)

    # 리사이즈 (모델 입력 크기에 맞춤)
    saturated_resized = cv2.resize(saturated, (img_size, img_size))
    binary_resized = cv2.resize(binary, (img_size, img_size))

    # 텐서 변환
    sat_tensor = torch.from_numpy(saturated_resized).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    bin_tensor = torch.from_numpy(binary_resized).float().unsqueeze(0) / 255.0          # (1, H, W)

    # 배치 차원 추가
    sat_tensor = sat_tensor.unsqueeze(0)  # (1, 3, H, W)
    bin_tensor = bin_tensor.unsqueeze(0)  # (1, 1, H, W)

    scale_info = {
        'orig_h': orig_h,
        'orig_w': orig_w,
        'img_size': img_size,
        'saturated_bgr': cv2.cvtColor(saturated, cv2.COLOR_RGB2BGR),
    }

    return sat_tensor, bin_tensor, original_bgr, scale_info


# ═══════════════════════════════════════════════════════════════════════
# 3. NMS (Non-Maximum Suppression)
# ═══════════════════════════════════════════════════════════════════════

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    YOLOv7 출력에서 NMS를 적용하여 최종 탐지 결과를 반환합니다.

    Args:
        prediction: (1, N, 85) — [x, y, w, h, obj_conf, cls1, cls2, ...]
        conf_thres: confidence threshold
        iou_thres: IoU threshold for NMS

    Returns:
        list of tensors, 각 (num_det, 6) — [x1, y1, x2, y2, conf, cls]
    """
    output = []

    for xi, x in enumerate(prediction):
        # Confidence 필터링
        obj_conf = x[:, 4]
        mask = obj_conf > conf_thres
        x = x[mask]

        if x.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue

        # cls_conf = obj_conf * cls_prob
        x[:, 5:] *= x[:, 4:5]  # class confidence = obj_conf * class_prob

        # xywh → xyxy
        boxes = xywh2xyxy(x[:, :4])

        # 가장 높은 class confidence & class index
        conf, cls_idx = x[:, 5:].max(dim=1, keepdim=True)

        # confidence threshold 재적용
        valid = conf.squeeze(-1) > conf_thres
        boxes = boxes[valid]
        conf = conf[valid]
        cls_idx = cls_idx[valid]

        if boxes.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue

        # 클래스별 NMS
        detections = torch.cat([boxes, conf, cls_idx.float()], dim=1)  # (N, 6)

        # torchvision NMS
        from torchvision.ops import nms
        keep = nms(detections[:, :4], detections[:, 4], iou_thres)
        detections = detections[keep]

        output.append(detections)

    return output


def xywh2xyxy(x):
    """중심좌표 (cx, cy, w, h) → 코너좌표 (x1, y1, x2, y2) 변환"""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


# ═══════════════════════════════════════════════════════════════════════
# 4. 결과 시각화
# ═══════════════════════════════════════════════════════════════════════

def draw_detections(image, detections, scale_info, class_names=None):
    """
    탐지 결과를 이미지 위에 바운딩박스 + 라벨로 시각화합니다.

    Args:
        image: 원본 BGR 이미지
        detections: (N, 6) tensor — [x1, y1, x2, y2, conf, cls]
        scale_info: 리사이즈 비율 정보
        class_names: 클래스 이름 리스트
    """
    if class_names is None:
        class_names = CURRENT_CLASS_NAMES

    img = image.copy()
    orig_h, orig_w = scale_info['orig_h'], scale_info['orig_w']
    img_size = scale_info['img_size']

    # 좌표를 원본 이미지 크기로 스케일링
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
        cls_id = int(cls_id)

        # 스케일 복원
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # 바운딩 박스 색상
        color = tuple(int(c) for c in COLORS[cls_id % len(COLORS)])

        # 바운딩 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 라벨 텍스트
        label = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 라벨 배경
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def create_comparison_image(original, saturated, fused_np, result):
    """
    원본 / Saturated / Fused / Detection 결과를 한 장에 비교하는 이미지를 생성합니다.
    """
    h = 300
    images = []
    labels = ['Original', 'Saturated', 'Fused', 'Detection']

    for img, label in zip([original, saturated, fused_np, result], labels):
        # 리사이즈
        aspect = img.shape[1] / img.shape[0]
        w = int(h * aspect)
        resized = cv2.resize(img, (w, h))

        # 라벨 추가
        cv2.putText(resized, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        images.append(resized)

    # 가로로 연결 (너비가 다를 수 있으므로 가장 좁은 너비에 맞춤)
    min_w = min(img.shape[1] for img in images)
    images = [img[:, :min_w] for img in images]
    comparison = np.vstack(images)

    return comparison


# ═══════════════════════════════════════════════════════════════════════
# 5. 메인 추론 파이프라인
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(image_path, fusion_model, yolo_model, device,
                  img_size=416, conf_thres=0.25, iou_thres=0.45,
                  save_dir=None, glare_L=None):
    """
    단일 이미지에 대해 전체 추론 파이프라인을 실행합니다.

    Pipeline:
        원본 이미지 → 고조도 시뮬레이션 → Saturated + Binary 생성
        → FusionNet → 융합 이미지 → YOLOv7 → NMS → 결과 시각화
    """
    start_time = time.time()

    # 1) 전처리
    sat_tensor, bin_tensor, original_bgr, scale_info = preprocess_image(image_path, img_size, glare_L=glare_L)
    sat_tensor = sat_tensor.to(device)
    bin_tensor = bin_tensor.to(device)

    # 2) Fusion
    fused = fusion_model(sat_tensor, bin_tensor)  # (1, 3, H, W)

    # 3) Fused 이미지를 numpy로 변환 (시각화용)
    fused_np = fused[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    fused_np = np.clip(fused_np * 255.0, 0, 255).astype(np.uint8)
    fused_np_bgr = cv2.cvtColor(fused_np, cv2.COLOR_RGB2BGR)
    fused_np_display = cv2.resize(fused_np_bgr, (scale_info['orig_w'], scale_info['orig_h']))

    # 4) YOLOv7 추론 (eval 모드에서는 (inference_out, train_out) 반환)
    yolo_out = yolo_model(fused)
    if isinstance(yolo_out, tuple):
        pred = yolo_out[0]  # inference output: (1, N, 85)
    else:
        pred = yolo_out

    # 5) NMS
    detections = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    det = detections[0]  # 배치 내 첫 번째 이미지

    elapsed = time.time() - start_time

    # 6) 결과 시각화
    result_img = draw_detections(original_bgr, det, scale_info)

    # 7) 비교 이미지 생성
    comparison = create_comparison_image(
        original_bgr,
        scale_info['saturated_bgr'],
        fused_np_display,
        result_img
    )

    # 8) 결과 출력
    num_det = det.shape[0]
    print(f"\n{'─' * 50}")
    print(f"  이미지: {os.path.basename(image_path)}")
    print(f"  추론 시간: {elapsed:.3f}s")
    print(f"  탐지된 객체: {num_det}개")

    if num_det > 0:
        for i in range(num_det):
            cls_id = int(det[i, 5])
            conf = det[i, 4].item()
            name = CURRENT_CLASS_NAMES[cls_id] if cls_id < len(CURRENT_CLASS_NAMES) else f"class_{cls_id}"
            print(f"    [{i+1}] {name}: {conf:.3f}")
    print(f"{'─' * 50}")

    # 9) 저장
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 개별 결과 저장
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_detection.jpg"), result_img)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_fused.jpg"), fused_np_display)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_comparison.jpg"), comparison)
        print(f"  결과 저장: {save_dir}/{base_name}_*.jpg")

    return result_img, det, fused_np_display


# ═══════════════════════════════════════════════════════════════════════
# 6. CLI 인터페이스
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="WDR Fusion + YOLOv7 추론 스크립트")

    # 입력
    parser.add_argument("--image", type=str, default=None,
                        help="단일 이미지 경로")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="이미지 폴더 경로 (폴더 내 모든 이미지 추론)")

    # 모델 가중치
    parser.add_argument("--fusion_ckpt", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_fusion_model.pth"),
                        help="FusionNet 체크포인트 경로")
    parser.add_argument("--yolo_weights", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_yolo_model.pth"),
                        help="YOLOv7 가중치 경로 (.pth state_dict 또는 .pt 전체 모델)")
    parser.add_argument("--yolo_cfg", type=str, default=None,
                        help="YOLOv7 config yaml 경로 (state_dict 로드 시 필수)")

    # 추론 파라미터
    parser.add_argument("--img_size", type=int, default=416,
                        help="모델 입력 이미지 크기 (default: 416)")
    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--glare_L", type=float, default=None,
                        help="고정 glare 강도 L (예: 0, 0.5, 2.0). 미지정 시 랜덤/기본값 사용")

    # 출력
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(config.OUTPUT_DIR, "inference_results"),
                        help="결과 저장 디렉토리")

    args = parser.parse_args()

    # ─── Validation ───
    if args.image is None and args.image_dir is None:
        parser.error("--image 또는 --image_dir 중 하나를 지정해야 합니다.")

    # ─── Device ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    # ─── 모델 로딩 ───
    print("\n=== 모델 로딩 ===")
    fusion_model = load_fusion_model(args.fusion_ckpt, device)
    yolo_model = load_yolo_model(args.yolo_weights, device, yolo_cfg=args.yolo_cfg)

    # ─── 추론 대상 이미지 수집 ───
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for f in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(f)[1].lower() in exts:
                image_paths.append(os.path.join(args.image_dir, f))

    if not image_paths:
        print("[!] 추론할 이미지가 없습니다.")
        return

    print(f"\n=== 추론 시작 ({len(image_paths)}장) ===")

    # ─── 추론 루프 ───
    all_detections = []
    for img_path in image_paths:
        try:
            _, det, _ = run_inference(
                img_path, fusion_model, yolo_model, device,
                img_size=args.img_size,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                save_dir=args.save_dir,
                glare_L=args.glare_L
            )
            all_detections.append((img_path, det))
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")

    # ─── 요약 ───
    print(f"\n{'═' * 50}")
    print(f"  추론 완료: {len(all_detections)}/{len(image_paths)}장")
    total_det = sum(d.shape[0] for _, d in all_detections)
    print(f"  총 탐지 객체: {total_det}개")
    print(f"  결과 저장 위치: {args.save_dir}")
    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    main()
