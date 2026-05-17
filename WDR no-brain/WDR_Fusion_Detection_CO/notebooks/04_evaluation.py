import torch
import sys
import os
import cv2
import numpy as np
from skimage.measure import shannon_entropy

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.fusion_net import FusionNet
from src.models.yolo_wrapper import YoloWrapper
from src import config

def compute_qabf(img1, img2, fused):
    """
    Q_abf (Edge Preservation Value) 계산 함수
    img1: 원본 이미지 (Original)
    img2: 이진 이미지 (Binary) - 논문의 WDR 문맥에서 융합 대상
    fused: 융합된 이미지 (Fused)
    (모두 numpy 배열, grayscale [0, 255] 기준)
    """
    def get_sobel_edges(img):
        # Sobel 필터 적용 (수평, 수직)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # 에지 강도(Magnitude) 및 방향(Orientation) 계산
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)
        return magnitude, orientation

    mag1, ori1 = get_sobel_edges(img1)
    mag2, ori2 = get_sobel_edges(img2)
    magF, oriF = get_sobel_edges(fused)
    
    # Q_abf 계산의 단순화된 근사 (실제 논문 수식의 복잡한 계수 생략, 핵심적인 에지 유사도 비교)
    # 융합된 이미지의 에지 강도가 원본/이진 이미지의 에지 강도와 얼마나 유사한지 비교
    weight1 = mag1 / (mag1 + mag2 + 1e-6)
    weight2 = mag2 / (mag1 + mag2 + 1e-6)
    
    # 강도 보존 비율 (Q_g)
    Qg1 = np.where(mag1 > magF, (magF + 1e-6) / (mag1 + 1e-6), (mag1 + 1e-6) / (magF + 1e-6))
    Qg2 = np.where(mag2 > magF, (magF + 1e-6) / (mag2 + 1e-6), (mag2 + 1e-6) / (magF + 1e-6))
    
    # 방향 보존 비율 (Q_a)
    Qa1 = 1.0 - np.abs(ori1 - oriF) / np.pi
    Qa2 = 1.0 - np.abs(ori2 - oriF) / np.pi
    
    # 최종 에지 보존 Q_abf
    Q1 = Qg1 * Qa1
    Q2 = Qg2 * Qa2
    
    Q_abf = np.sum(Q1 * weight1 + Q2 * weight2) / np.sum(weight1 + weight2)
    return float(Q_abf)

def tensor_to_gray_numpy(tensor):
    """ 텐서 (B, C, H, W) -> numpy grayscale (H, W) 반환 (배치의 첫 번째 이미지 기준) """
    # 정규화 해제 후 [0, 255] uint8 변환
    img = tensor[0].cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.shape[2] == 1:
        img = img.squeeze(-1)
    return img

def compute_map_metrics(preds, targets):
    """
    mAP@50:95 계산 (간단한 COCO-style 근사 구현)
    - preds: (B, N, 5+nc) 형태의 YOLOv7 inference output (xywh, obj_conf, cls_conf...)
    - targets: (M, 6) 형태 [batch_idx, class, cx, cy, w, h] (normalized)
    """
    import math

    def xywh2xyxy(xywh):
        # xywh: (..., 4) in pixels
        x, y, w, h = xywh.unbind(-1)
        return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)

    def box_iou(box1, box2):
        # box1: (N,4), box2: (M,4) in xyxy
        area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
        area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter + 1e-9
        return inter / union

    def nms_pytorch(boxes, scores, iou_thres=0.45):
        try:
            from torchvision.ops import nms as tv_nms
            return tv_nms(boxes, scores, iou_thres)
        except Exception:
            # pure torch NMS fallback
            idxs = scores.argsort(descending=True)
            keep = []
            while idxs.numel() > 0:
                i = idxs[0]
                keep.append(i)
                if idxs.numel() == 1:
                    break
                ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
                idxs = idxs[1:][ious <= iou_thres]
            return torch.stack(keep) if keep else boxes.new_zeros((0,), dtype=torch.long)

    def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45):
        # pred: (B, N, 5+nc) where 0:4 xywh, 4 obj, 5: cls conf
        out = []
        for p in pred:
            if p.numel() == 0:
                out.append(p.new_zeros((0, 6)))
                continue
            obj = p[:, 4]
            cls_conf, cls_idx = p[:, 5:].max(dim=1)
            conf = obj * cls_conf
            m = conf > conf_thres
            p = p[m]
            conf = conf[m]
            cls_idx = cls_idx[m]
            if p.shape[0] == 0:
                out.append(p.new_zeros((0, 6)))
                continue
            boxes = xywh2xyxy(p[:, :4])
            keep = nms_pytorch(boxes, conf, iou_thres=iou_thres)
            det = torch.cat([boxes[keep], conf[keep, None], cls_idx[keep, None].float()], dim=1)  # (x1,y1,x2,y2,conf,cls)
            out.append(det)
        return out

    def compute_ap(recall, precision):
        # COCO-style area under PR curve (interpolated)
        mrec = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
        mpre = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])
        for i in range(mpre.numel() - 1, 0, -1):
            mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
        idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze(1)
        return torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item()

    device = preds.device if isinstance(preds, torch.Tensor) else torch.device("cpu")
    if not isinstance(preds, torch.Tensor):
        return 0.0

    # Build GT by image index
    # targets: (M,6) [batch, cls, cx,cy,w,h] normalized
    gt_by_img = {}
    if targets is not None and targets.numel() > 0:
        for row in targets.detach().to("cpu"):
            bi = int(row[0].item())
            cls = int(row[1].item())
            gt_by_img.setdefault(bi, []).append((cls, row[2:].clone()))

    detections = non_max_suppression(preds.detach().to("cpu"), conf_thres=0.25, iou_thres=0.45)

    # Determine image size from prediction scale heuristics (we can't reliably read it from preds alone)
    # Assume training/eval uses config.IMAGE_SIZE
    img_w, img_h = getattr(config, "IMAGE_SIZE", (416, 416))

    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50..0.95

    # Accumulate stats: per class, per iou threshold
    ap_all = []
    # classes present in GT or preds
    present_classes = set()
    for img_i, gts in gt_by_img.items():
        for cls, _ in gts:
            present_classes.add(cls)
    for det in detections:
        if det.numel() > 0:
            present_classes |= set(det[:, 5].int().tolist())
    present_classes = sorted(list(present_classes))
    if len(present_classes) == 0:
        return 0.0

    for cls in present_classes:
        # gather all predictions of this class across batch
        pred_list = []
        gt_count = 0
        for img_i, det in enumerate(detections):
            gts = gt_by_img.get(img_i, [])
            gt_cls = [gt for gt in gts if gt[0] == cls]
            gt_count += len(gt_cls)

            if det.numel() == 0:
                continue
            det_cls = det[det[:, 5].int() == cls]
            if det_cls.numel() == 0:
                continue
            for d in det_cls:
                pred_list.append((img_i, float(d[4].item()), d[:4].clone()))

        if gt_count == 0:
            continue

        if len(pred_list) == 0:
            ap_all.extend([0.0] * len(iou_thresholds))
            continue

        pred_list.sort(key=lambda x: x[1], reverse=True)
        pred_imgs = [p[0] for p in pred_list]
        pred_confs = torch.tensor([p[1] for p in pred_list])
        pred_boxes = torch.stack([p[2] for p in pred_list], dim=0)

        # Precompute GT boxes per image for this class (xyxy pixels)
        gt_boxes_by_img = {}
        for img_i, gts in gt_by_img.items():
            gt_cls = [gt for gt in gts if gt[0] == cls]
            if not gt_cls:
                continue
            boxes_norm = torch.stack([gt[1] for gt in gt_cls], dim=0)  # (K,4) cxcywh norm
            # to pixels xyxy
            cx = boxes_norm[:, 0] * img_w
            cy = boxes_norm[:, 1] * img_h
            w = boxes_norm[:, 2] * img_w
            h = boxes_norm[:, 3] * img_h
            boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
            gt_boxes_by_img[img_i] = boxes_xyxy

        for iou_t in iou_thresholds:
            tp = torch.zeros((len(pred_list),), dtype=torch.float32)
            fp = torch.zeros((len(pred_list),), dtype=torch.float32)
            matched = {img_i: torch.zeros((gt_boxes_by_img.get(img_i, torch.empty((0, 4))).shape[0],), dtype=torch.bool)
                       for img_i in gt_boxes_by_img.keys()}

            for i, img_i in enumerate(pred_imgs):
                gt_boxes = gt_boxes_by_img.get(img_i, None)
                if gt_boxes is None or gt_boxes.numel() == 0:
                    fp[i] = 1.0
                    continue
                ious = box_iou(pred_boxes[i].unsqueeze(0), gt_boxes).squeeze(0)
                j = int(torch.argmax(ious).item())
                best_iou = float(ious[j].item())
                if best_iou >= iou_t and not matched[img_i][j]:
                    tp[i] = 1.0
                    matched[img_i][j] = True
                else:
                    fp[i] = 1.0

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            recall = tp_cum / (gt_count + 1e-9)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap_all.append(compute_ap(recall, precision))

    if len(ap_all) == 0:
        return 0.0
    return float(sum(ap_all) / len(ap_all))

def evaluate_pipeline(fusion_model, yolo_model, dataloader):
    device = config.DEVICE
    fusion_model.eval()
    yolo_wrapper = YoloWrapper(yolo_model).to(device)
    yolo_wrapper.eval()
    
    total_entropy_fused = 0
    total_entropy_sat = 0
    total_qabf = 0
    map_scores = []
    
    num_batches = len(dataloader)
    if num_batches == 0:
        print("Dataloader is empty.")
        return 0, 0, 0

    print("Evaluating pipeline...")
    with torch.no_grad():
        for batch_idx, (original, saturated, binary, targets) in enumerate(dataloader):
            original, saturated, binary, targets = original.to(device), saturated.to(device), binary.to(device), targets.to(device)
            
            # Fused image 생성 시 saturated 텐서를 전달
            fused = fusion_model(saturated, binary)
            
            # --- 1. 정보량 평가 (Entropy) ---
            fused_np = tensor_to_gray_numpy(fused)
            sat_np = tensor_to_gray_numpy(saturated)
            
            # 픽셀 정보량(Entropy) 측정
            ent_fused = shannon_entropy(fused_np)
            ent_sat = shannon_entropy(sat_np)
            
            total_entropy_fused += ent_fused
            total_entropy_sat += ent_sat
            
            # --- 2. 에지 보존력 평가 (Q_abf) ---
            orig_np = tensor_to_gray_numpy(original)
            bin_np = tensor_to_gray_numpy(binary)
            
            qabf_val = compute_qabf(orig_np, bin_np, fused_np)
            total_qabf += qabf_val
            
            # --- 3. 객체 탐지 성능 평가 (mAP) ---
            preds = yolo_wrapper(fused)
            map_val = compute_map_metrics(preds, targets)
            map_scores.append(map_val)
            
    # 통계 평균
    avg_entropy_fused = total_entropy_fused / num_batches
    avg_entropy_sat = total_entropy_sat / num_batches
    avg_qabf = total_qabf / num_batches
    final_map = sum(map_scores) / len(map_scores)
    
    print("\n=== Evaluation Results ===")
    print(f"1. Information (Entropy): Fused = {avg_entropy_fused:.4f} (Saturated = {avg_entropy_sat:.4f})")
    print(f"2. Edge Preservation (Q_abf): {avg_qabf:.4f}")
    print(f"3. Detection mAP@50:95: {final_map:.4f}\n")
    
    return avg_entropy_fused, avg_qabf, final_map

if __name__ == "__main__":
    print("Evaluation script ready.")
