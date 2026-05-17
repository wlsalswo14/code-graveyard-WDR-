import torch
import torch.nn.functional as F

def intensity_loss(fused_image, original_image):
    """L1 Loss between fused image and original image"""
    return F.l1_loss(fused_image, original_image)
