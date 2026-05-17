import torch
import torch.nn as nn

class YoloWrapper(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        
        # 📌 YOLOv7의 fuse()로 인해 발생한 Non-leaf tensor를 Leaf tensor로 변환
        # (이로 인해 두 번째 에폭에서 RuntimeError: Trying to backward through the graph a second time 발생 방지)
        for module in yolo_model.modules():
            if hasattr(module, 'weight') and module.weight is not None and not module.weight.is_leaf:
                module.weight = nn.Parameter(module.weight.detach())
            if hasattr(module, 'bias') and module.bias is not None and not module.bias.is_leaf:
                module.bias = nn.Parameter(module.bias.detach())
                
        self.model = yolo_model
        
    def forward(self, x, force_train_out=False):
        out = self.model(x)
        
        # YOLOv7 returns (inference_out, train_out) when in eval mode
        # If force_train_out is True, we return train_out even in eval mode.
        if isinstance(out, tuple) and len(out) == 2:
            inference_out, train_out = out
            if force_train_out:
                return train_out
            return train_out if self.model.training else inference_out
        
        return out

    def freeze(self):
        """Freeze YOLO weights and BatchNorm stats for Stage 1 Fusion training"""
        self.model.eval()  # Set to eval mode to freeze BN stats and Dropout
        
        # Ensure all BN layers are in eval mode even if self.model.train() is called later
        # (Though we should avoid calling train() on the wrapper during Stage 1)
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm)):
                module.eval()
                
        for param in self.model.parameters():
            if param.is_leaf:
                param.requires_grad = False
            
    def unfreeze_head_only(self):
        """Unfreeze ONLY YOLO detector heads for Stage 2 initial stabilization"""
        self.model.train()
        print("Selective unfreezing: Freezing everything EXCEPT Detector Heads.")
        
        for name, param in self.model.named_parameters():
            if not param.is_leaf:
                continue
                
            # YOLOv7-tiny Detector Head 판별
            # 1. IDetect 계층 (최종 출력 헤드)
            # 2. tiny 모델의 하단 레이어 인덱스 (보통 70번대 이후)
            is_head = 'IDetect' in name or any(f'.{i}.' in name for i in range(70, 100))
            param.requires_grad = is_head

    def unfreeze_selective(self, threshold=27):
        """Unfreeze YOLO weights selectively (index >= threshold)"""
        self.model.train()
        print(f"Selective unfreezing: Freezing Backbone (index < {threshold}), Unfreezing Neck/Head.")
        
        for name, param in self.model.named_parameters():
            if not param.is_leaf:
                continue
                
            try:
                parts = name.split('.')
                if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx < threshold:
                        param.requires_grad = False
                        continue
            except Exception:
                pass
                
            param.requires_grad = True

    def unfreeze(self):
        """Legacy compatibility: Default unfreeze to Neck+Head (index >= 27 for tiny)"""
        self.unfreeze_selective(threshold=27)
