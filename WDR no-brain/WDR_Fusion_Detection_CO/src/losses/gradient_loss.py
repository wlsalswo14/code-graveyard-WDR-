import torch
import torch.nn.functional as F

def sobel_filter(image):
    """
    Apply Sobel filter to extract image gradients.
    Assumes image is (B, C, H, W).
    """
    device = image.device
    channels = image.shape[1]
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    
    # Apply padding
    pad = (1, 1, 1, 1)
    padded_img = F.pad(image, pad, mode='reflect')
    
    # Calculate gradients
    grad_x = F.conv2d(padded_img, sobel_x, groups=channels)
    grad_y = F.conv2d(padded_img, sobel_y, groups=channels)
    
    # Magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    return grad_mag

def gradient_loss(fused_image, original_image):
    """
    Paper Eq 6, 7: Gradient Loss comparing ONLY fused and original images.
    Binary image is NOT used here.
    """
    sobel_fused = sobel_filter(fused_image)
    sobel_orig = sobel_filter(original_image)
    
    return F.l1_loss(sobel_fused, sobel_orig)
