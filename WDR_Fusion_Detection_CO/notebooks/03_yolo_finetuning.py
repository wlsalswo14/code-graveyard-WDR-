import torch
import torch.optim as optim
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "yolov7"))
from src.models.yolo_wrapper import YoloWrapper
from src import config

def finetune_yolo(fusion_model, yolo_model, dataloader, epochs=50):
    device = config.DEVICE
    fusion_model.eval() # Stage 2: Fusion 모델은 동결
    
    yolo_wrapper = YoloWrapper(yolo_model).to(device)
    yolo_wrapper.unfreeze() # YOLO 모델 학습
    
    from utils.loss import ComputeLoss
    compute_loss = ComputeLoss(yolo_wrapper.model)
    
    optimizer = optim.Adam(yolo_wrapper.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        yolo_wrapper.train()
        total_loss = 0
        
        # [수정 1] dataloader에서 반환하는 4개의 변수를 모두 받습니다.
        for batch_idx, (original, saturated, binary, targets) in enumerate(dataloader):
            # [수정 2] 4개의 변수를 모두 device로 보냅니다.
            original, saturated, binary, targets = original.to(device), saturated.to(device), binary.to(device), targets.to(device)
            
            # [수정 3] Fusion 모델의 입력으로 원본(original)이 아닌 포화된 이미지(saturated)를 사용합니다.
            with torch.no_grad():
                fused = fusion_model(saturated, binary)
            
            optimizer.zero_grad()
            yolo_output = yolo_wrapper(fused)
            
            # YOLOv7 실제 탐지 손실
            loss, loss_items = compute_loss(yolo_output, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], YOLO Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    print("Stage 2 YOLO Finetuning script ready.")
