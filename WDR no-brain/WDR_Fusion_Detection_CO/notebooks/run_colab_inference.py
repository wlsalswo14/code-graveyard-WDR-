"""
run_colab_inference.py — FusionNet + YOLOv7 추론 (로컬·Colab 공통)

예시 (프로젝트 루트에서):

    python notebooks/run_colab_inference.py --image path/to/test.jpg

    python notebooks/run_colab_inference.py --image test.jpg \\
        --yolo_ckpt output/checkpoints/best_yolo_model.pth

COCO 사전학습 yolov7-tiny.pt 는 프로젝트 루트·weights/ 등에 두거나
환경변수 YOLO_PRETRAINED_PATH 로 지정합니다.
"""

import os
import sys
import torch

# PyTorch 2.6+ 호환성 문제 해결 (weights_only 기본값 변경 방지)
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

import cv2
import numpy as np
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# ─── 경로 설정 ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "yolov7"))

from src.models.fusion_net import FusionNet
from src import config
from src.data.augmentation import (
    simulate_high_glare,
    generate_binary_from_saturated,
    compute_threshold_prm,
)

# ─── COCO 클래스 이름 ──────────────────────────────────────────────
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

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_NAMES), 3), dtype=np.uint8)


def xywh2xyxy(x):
    """중심좌표 → 코너좌표 변환"""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """NMS 적용"""
    from torchvision.ops import nms
    output = []

    for x in prediction:
        mask = x[:, 4] > conf_thres
        x = x[mask]

        if x.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue

        x[:, 5:] *= x[:, 4:5]
        boxes = xywh2xyxy(x[:, :4])
        conf, cls_idx = x[:, 5:].max(dim=1, keepdim=True)

        valid = conf.squeeze(-1) > conf_thres
        boxes, conf, cls_idx = boxes[valid], conf[valid], cls_idx[valid]

        if boxes.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue

        detections = torch.cat([boxes, conf, cls_idx.float()], dim=1)
        keep = nms(detections[:, :4], detections[:, 4], iou_thres)
        output.append(detections[keep])

    return output


@torch.no_grad()
def run_inference(image_path, fusion_ckpt, yolo_ckpt=None,
                  yolo_cfg="yolov7/cfg/training/yolov7-tiny.yaml",
                  img_size=416, conf_thres=0.15, iou_thres=0.45):
    """
    전체 추론 파이프라인 실행 + matplotlib 시각화

    Args:
        image_path:   추론할 이미지 경로
        fusion_ckpt:  FusionNet 체크포인트 (.pth)
        yolo_ckpt:    파인튜닝된 YOLOv7 체크포인트 (.pth, 선택사항)
        yolo_cfg:     YOLOv7 config yaml 경로
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ═══════════════════════════════════════════════════════════════
    # 1. 모델 로딩
    # ═══════════════════════════════════════════════════════════════
    print("\n📦 모델 로딩 중...")

    # FusionNet
    fusion_model = FusionNet(out_channels=3).to(device)
    if os.path.exists(fusion_ckpt):
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        print(f"  ✅ FusionNet 로드 완료: {fusion_ckpt}")
    else:
        print(f"  ⚠️  체크포인트 없음: {fusion_ckpt}")
        print(f"     랜덤 가중치로 추론합니다 (결과가 좋지 않을 수 있음)")
    fusion_model.eval()

    # YOLOv7 — COCO 사전학습 모델을 기본으로 로드
    from models.experimental import attempt_load
    
    pretrained_path = config.find_yolov7_pretrained("yolov7-tiny.pt")

    if pretrained_path is None:
        print("  ❌ COCO 사전학습 가중치(yolov7-tiny.pt)를 찾을 수 없습니다.")
        print("     프로젝트 루트나 weights/에 배치하거나 YOLO_PRETRAINED_PATH를 설정하세요.")
        print("     https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt")
        return None, None
    
    yolo_model = attempt_load(pretrained_path, map_location=device)
    print(f"  ✅ YOLOv7 COCO 사전학습 모델 로드: {pretrained_path}")

    # 파인튜닝된 가중치가 있으면 덮어쓰기
    if yolo_ckpt and os.path.exists(yolo_ckpt):
        yolo_model.load_state_dict(torch.load(yolo_ckpt, map_location=device))
        print(f"  ✅ YOLOv7 파인튜닝 가중치 로드: {yolo_ckpt}")
    else:
        print(f"  ℹ️  파인튜닝 가중치 없음 → COCO 사전학습 가중치로 추론")
    yolo_model.eval()

    # ═══════════════════════════════════════════════════════════════
    # 2. 이미지 전처리
    # ═══════════════════════════════════════════════════════════════
    print("\n🖼️  이미지 전처리 중...")

    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = original_rgb.shape[:2]
    print(f"  원본 크기: {orig_w} x {orig_h}")

    # 고조도 시뮬레이션 → Saturated + Binary
    saturated_rgb, raw_irradiance = simulate_high_glare(original_rgb)
    threshold = compute_threshold_prm(saturated_rgb)
    binary = generate_binary_from_saturated(raw_irradiance, threshold=threshold)
    print(f"  동적 임계값: {threshold}")

    # 리사이즈 + 텐서 변환
    sat_resized = cv2.resize(saturated_rgb, (img_size, img_size))
    bin_resized = cv2.resize(binary, (img_size, img_size))

    sat_tensor = torch.from_numpy(sat_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    bin_tensor = torch.from_numpy(bin_resized).float().unsqueeze(0).unsqueeze(0) / 255.0

    sat_tensor = sat_tensor.to(device)
    bin_tensor = bin_tensor.to(device)

    # ═══════════════════════════════════════════════════════════════
    # 3. FusionNet 추론
    # ═══════════════════════════════════════════════════════════════
    print("\n🔥 FusionNet 추론 중...")
    t0 = time.time()

    fused = fusion_model(sat_tensor, bin_tensor)  # (1, 3, H, W)

    fusion_time = time.time() - t0
    print(f"  Fusion 완료: {fusion_time:.3f}s")

    # Fused 이미지 numpy 변환
    fused_np = fused[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3) RGB
    fused_np = np.clip(fused_np * 255.0, 0, 255).astype(np.uint8)

    # ═══════════════════════════════════════════════════════════════
    # 4. YOLOv7 탐지
    # ═══════════════════════════════════════════════════════════════
    print("\n🎯 YOLOv7 탐지 중...")
    t1 = time.time()

    yolo_out = yolo_model(fused)
    if isinstance(yolo_out, tuple):
        pred = yolo_out[0]  # inference output
    else:
        pred = yolo_out

    detections = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    det = detections[0]

    detect_time = time.time() - t1
    print(f"  탐지 완료: {detect_time:.3f}s")
    print(f"  탐지된 객체: {det.shape[0]}개")

    # ═══════════════════════════════════════════════════════════════
    # 5. 탐지 결과 출력
    # ═══════════════════════════════════════════════════════════════
    if det.shape[0] > 0:
        print("\n📋 탐지 결과:")
        print(f"  {'#':<4} {'클래스':<15} {'신뢰도':<10} {'위치 (x1,y1,x2,y2)'}")
        print(f"  {'─' * 55}")
        for i in range(det.shape[0]):
            x1, y1, x2, y2, conf, cls_id = det[i].cpu().numpy()
            cls_id = int(cls_id)
            name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"cls_{cls_id}"
            print(f"  {i+1:<4} {name:<15} {conf:<10.3f} ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    else:
        print("\n  ⚠️  탐지된 객체가 없습니다. conf_thres를 낮춰보세요.")

    # ═══════════════════════════════════════════════════════════════
    # 6. 결과 시각화 (matplotlib)
    # ═══════════════════════════════════════════════════════════════
    print("\n📊 결과 시각화...")

    # 바운딩박스를 그릴 이미지 (fused 이미지 기준)
    fused_display = fused_np.copy()
    scale_x = img_size / img_size  # 1.0 (fused는 이미 img_size)
    scale_y = img_size / img_size

    for i in range(det.shape[0]):
        x1, y1, x2, y2, conf, cls_id = det[i].cpu().numpy()
        cls_id = int(cls_id)
        color = tuple(int(c) for c in COLORS[cls_id % len(COLORS)])
        name = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"cls_{cls_id}"

        cv2.rectangle(fused_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(fused_display, (int(x1), int(y1)-th-4), (int(x1)+tw, int(y1)), color, -1)
        cv2.putText(fused_display, label, (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Saturated 이미지 리사이즈 (비교용)
    sat_display = cv2.resize(saturated_rgb, (img_size, img_size))
    bin_display = cv2.resize(binary, (img_size, img_size))

    # Original 리사이즈
    orig_display = cv2.resize(original_rgb, (img_size, img_size))

    # ─── 4장 비교 그래프 ───
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(orig_display)
    axes[0].set_title("① Original", fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(sat_display)
    axes[1].set_title("② Saturated (고조도)", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(fused_np)
    axes[2].set_title("③ Fused (복원)", fontsize=13, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(fused_display)
    axes[3].set_title(f"④ Detection ({det.shape[0]}개)", fontsize=13, fontweight='bold')
    axes[3].axis('off')

    plt.suptitle("WDR Fusion + YOLOv7 추론 결과", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # ─── Binary 이미지도 별도로 표시 ───
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

    axes2[0].imshow(bin_display, cmap='gray')
    axes2[0].set_title("Binary Image (WDR)", fontsize=13, fontweight='bold')
    axes2[0].axis('off')

    axes2[1].imshow(fused_display)
    axes2[1].set_title("최종 탐지 결과", fontsize=13, fontweight='bold')
    axes2[1].axis('off')

    plt.tight_layout()
    plt.show()

    # ═══════════════════════════════════════════════════════════════
    # 7. 결과 파일 저장
    # ═══════════════════════════════════════════════════════════════
    save_dir = os.path.join(PROJECT_ROOT, "output", "inference_results")
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # BGR 변환 후 저장 (cv2는 BGR 사용)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_fused.jpg"),
                cv2.cvtColor(fused_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_detection.jpg"),
                cv2.cvtColor(fused_display, cv2.COLOR_RGB2BGR))

    print(f"\n💾 결과 저장 완료: {save_dir}/")
    print(f"   - {base_name}_fused.jpg")
    print(f"   - {base_name}_detection.jpg")

    print(f"\n{'═' * 50}")
    print(f"  총 추론 시간: {fusion_time + detect_time:.3f}s")
    print(f"  (Fusion: {fusion_time:.3f}s + Detection: {detect_time:.3f}s)")
    print(f"{'═' * 50}")

    return fused_np, det


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WDR Fusion + YOLOv7 추론")
    parser.add_argument("--image", type=str, required=True,
                        help="추론할 이미지 경로")
    parser.add_argument("--fusion_ckpt", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_fusion_model.pth"),
                        help="FusionNet 체크포인트 경로")
    parser.add_argument("--yolo_ckpt", type=str, default=None,
                        help="파인튜닝된 YOLOv7 체크포인트 경로 (.pth)")
    parser.add_argument("--yolo_cfg", type=str,
                        default=os.path.join(PROJECT_ROOT, "yolov7", "cfg", "training", "yolov7-tiny.yaml"),
                        help="YOLOv7 config yaml 경로")
    parser.add_argument("--conf_thres", type=float, default=0.15,
                        help="Confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.45,
                        help="IoU threshold")

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        fusion_ckpt=args.fusion_ckpt,
        yolo_ckpt=args.yolo_ckpt,
        yolo_cfg=args.yolo_cfg,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )
