from .intensity_loss import intensity_loss
from .gradient_loss import gradient_loss

def compute_total_loss(fused, original, yolo_output, targets, yolo_loss_fn, alpha=1.0, beta=1.0, gamma=0.5):
    """
    Combined Loss for Stage 1 Fusion Training.
    Paper: L_total = 1.0 * L_int + 1.0 * L_grad + 0.5 * L_det
    """
    l_int = intensity_loss(fused, original)
    l_grad = gradient_loss(fused, original)
    
    # detection loss computation
    if yolo_output is not None and targets is not None and yolo_loss_fn is not None:
        l_det_sum, _ = yolo_loss_fn(yolo_output, targets) 
        # YOLOv7 loss is sum over batch size, convert it to mean
        batch_size = fused.shape[0]
        l_det = l_det_sum / batch_size
    else:
        # If no YOLO is provided or we are evaluating fusion only
        l_det = 0.0
        
    l_total = alpha * l_int + beta * l_grad + gamma * l_det
    
    return l_total, l_int, l_grad, l_det
