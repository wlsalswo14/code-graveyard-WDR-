import torch

# PyTorch 2.6+ 호환성 문제 해결 (weights_only 기본값 변경 방지)
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "yolov7"))

from src.models.fusion_net import FusionNet
from src.models.yolo_wrapper import YoloWrapper
from src.losses.combined_loss import compute_total_loss
from src import config

def _set_dataset_epoch(dataloader, epoch: int):
    # random_split 결과는 Subset이므로 원본 dataset까지 내려가며 설정
    try:
        ds = dataloader.dataset
        while hasattr(ds, "dataset"):
            ds = ds.dataset
        if hasattr(ds, "current_epoch"):
            ds.current_epoch = int(epoch)
    except Exception:
        pass

def train_fusion_model(train_dataloader, val_dataloader, yolo_model, epochs=config.EPOCHS):
    """
    Stage 1: Train Fusion Model with Validation and Checkpoint Saving
    - train_dataloader와 val_dataloader를 분리하여 입력받습니다.
    """
    # config에 정의된 DEVICE가 없으면 기본값(cuda 또는 cpu) 지정
    device = getattr(config, 'DEVICE', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    fusion_model = FusionNet(out_channels=3).to(device)
    yolo_wrapper = YoloWrapper(yolo_model).to(device)
    
    # 📌 [최적화] PyTorch 2.0+ Compile 적용
    if getattr(config, 'USE_COMPILE', False) and hasattr(torch, 'compile'):
        try:
            print("Compiling models for faster execution...")
            fusion_model = torch.compile(fusion_model)
            # YOLO 모델은 구조에 따라 컴파일이 실패할 수 있으므로 선택적용 가능
            # yolo_wrapper = torch.compile(yolo_wrapper)
        except Exception as e:
            print(f"Compilation failed: {e}. Proceeding without compile.")
    
    # Stage 1: YOLO 모델의 파라미터는 동결(Freeze)
    yolo_wrapper.freeze()
    
    # YOLOv7의 Loss 계산 모듈
    from utils.loss import ComputeLoss
    
    # ComputeLoss 초기화를 위한 하이퍼파라미터 및 gr 값 설정 (기본값)
    if not hasattr(yolo_model, 'hyp'):
        yolo_model.hyp = {'box': 0.05, 'obj': 1.0, 'cls': 0.5, 'anchor_t': 4.0, 'cls_pw': 1.0, 'obj_pw': 1.0, 'fl_gamma': 0.0, 'label_smoothing': 0.0}
    if not hasattr(yolo_model, 'gr'):
        yolo_model.gr = 1.0
        
    compute_loss = ComputeLoss(yolo_model)
    
    optimizer = optim.AdamW(fusion_model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    accumulation_steps = max(1, int(getattr(config, "ACCUMULATION_STEPS", 1)))
    # OneCycleLR용 total_steps 계산 (배치/누적 스텝 고려)
    total_steps = epochs * ((len(train_dataloader) + accumulation_steps - 1) // accumulation_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, total_steps=total_steps, pct_start=0.3)
    use_amp = bool(getattr(config, "USE_AMP", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    
    # 📌 [수정] 체크포인트 저장을 위한 디렉토리 생성 및 베스트 지표 초기화
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    # 📌 [수정] Top-3 체크포인트 저장을 위한 리스트 (loss, path)
    best_fusion_checkpoints = []

    for epoch in range(epochs):
        _set_dataset_epoch(train_dataloader, epoch)
        _set_dataset_epoch(val_dataloader, epoch)
        warmup_epochs = int(getattr(config, "WARMUP_EPOCHS", 10))
        ramp_epochs = int(getattr(config, "GAMMA_RAMP_EPOCHS", 0))
        if epoch < warmup_epochs:
            gamma = 0.0
        elif ramp_epochs > 0 and epoch < (warmup_epochs + ramp_epochs):
            # 안정화를 위해 step 대신 ramp로 gamma를 점진적으로 증가
            progress = (epoch - warmup_epochs + 1) / ramp_epochs  # 1/ramp ... 1
            gamma = float(config.GAMMA) * float(progress)
        else:
            gamma = float(config.GAMMA)  # 0.5
        # -------------------------
        # 1. Training Phase
        # -------------------------
        fusion_model.train()
        total_train_loss = 0
        
        for batch_idx, (original, saturated, binary, targets) in enumerate(train_dataloader):
            original, saturated, binary, targets = original.to(device), saturated.to(device), binary.to(device), targets.to(device)
            
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
            with autocast_ctx:
                fused = fusion_model(saturated, binary)
                # Warm-up 단계에서는 detection-aware를 끄고(fusion-only) 안정화
                if gamma > 0:
                    yolo_output = yolo_wrapper(fused, force_train_out=True)
                    yolo_loss_fn = compute_loss
                    det_targets = targets
                else:
                    yolo_output = None
                    yolo_loss_fn = None
                    det_targets = None
                
                loss, l_int, l_grad, l_det = compute_total_loss(
                    fused, original, yolo_output, det_targets, yolo_loss_fn,
                    alpha=config.ALPHA, beta=config.BETA, gamma=gamma
                )
                loss = loss / accumulation_steps
            
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 📌 [수정] GradScaler AssertionError 방지: gradient가 None인지 확인
            has_grad = False
            for param in fusion_model.parameters():
                if param.grad is not None:
                    has_grad = True
                    break
                    
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_dataloader))
            if is_update_step:
                if has_grad:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # 📌 배치(업데이트) 단위로 스케줄러 이동
                else:
                    print(f"[Warning] Epoch {epoch}, Batch {batch_idx}: No gradients computed for fusion_model. Skipping optimizer step.")
            
            total_train_loss += loss.item() * accumulation_steps
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # -------------------------
        # 2. Validation Phase 
        # -------------------------
        fusion_model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for val_batch_idx, (v_original, v_saturated, v_binary, v_targets) in enumerate(val_dataloader):
                v_original, v_saturated, v_binary, v_targets = v_original.to(device), v_saturated.to(device), v_binary.to(device), v_targets.to(device)
                
                autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
                with autocast_ctx:
                    v_fused = fusion_model(v_saturated, v_binary)
                    if gamma > 0:
                        v_yolo_output = yolo_wrapper(v_fused, force_train_out=True)
                        v_yolo_loss_fn = compute_loss
                        v_det_targets = v_targets
                    else:
                        v_yolo_output = None
                        v_yolo_loss_fn = None
                        v_det_targets = None
                    
                    v_loss, _, _, _ = compute_total_loss(
                        v_fused, v_original, v_yolo_output, v_det_targets, v_yolo_loss_fn,
                        alpha=config.ALPHA, beta=config.BETA, gamma=gamma
                    )
                    total_val_loss += v_loss.item()
                    
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # -------------------------
        # 3. Top-3 Checkpoint Saving Logic
        # -------------------------
        save_start_epoch = int(getattr(config, "WARMUP_EPOCHS", 10)) + int(getattr(config, "GAMMA_RAMP_EPOCHS", 5))
        
        if epoch >= save_start_epoch:
            if len(best_fusion_checkpoints) < 3 or avg_val_loss < best_fusion_checkpoints[-1][0]:
                # 새 체크포인트 저장
                temp_path = os.path.join(config.CHECKPOINT_DIR, f'fusion_model_epoch_{epoch+1}_loss_{avg_val_loss:.4f}.pth')
                torch.save(fusion_model.state_dict(), temp_path)
                best_fusion_checkpoints.append((avg_val_loss, temp_path))
                best_fusion_checkpoints.sort(key=lambda x: x[0])
                
                # 상위 3개 초과 시 삭제
                if len(best_fusion_checkpoints) > 3:
                    _, old_path = best_fusion_checkpoints.pop(-1)
                    if os.path.exists(old_path):
                        os.remove(old_path)
                
                # 순위별 고정 파일명으로 복사/저장 (선택 사항: 관리 편의용)
                for i, (loss, path) in enumerate(best_fusion_checkpoints):
                    rank_path = os.path.join(config.CHECKPOINT_DIR, f'best_fusion_model_rank{i+1}.pth')
                    import shutil
                    shutil.copy(path, rank_path)
                
                print(f"  [*] Top-3 Fusion Checkpoints updated! Current Best Val Loss: {best_fusion_checkpoints[0][0]:.4f}")

    # 학습이 끝나면 가장 성능이 좋았던 가중치로 복구하여 반환
    if best_fusion_checkpoints:
        best_weights_path = os.path.join(config.CHECKPOINT_DIR, 'best_fusion_model_rank1.pth')
        if os.path.exists(best_weights_path):
            fusion_model.load_state_dict(torch.load(best_weights_path))
            print("Loaded Top-1 Fusion checkpoint for final return.")
        
    return fusion_model
def finetune_yolo_model(train_dataloader, fusion_model, yolo_wrapper, val_dataloader=None, epochs=config.EPOCHS):
    """
    Stage 2: Fine-tune YOLOv7 on fused images with Validation and Checkpoint Saving
    - val_dataloader가 None이면 train_dataloader를 val에도 사용합니다.
    """
    device = getattr(config, 'DEVICE', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    if val_dataloader is None:
        val_dataloader = train_dataloader
    
    # 📌 [최적화] PyTorch 2.0+ Compile 적용
    if getattr(config, 'USE_COMPILE', False) and hasattr(torch, 'compile'):
        try:
            print("Compiling models for faster execution...")
            # 이미 컴파일되었을 수도 있으므로 체크 후 진행
            if not hasattr(fusion_model, '_compiled_code'): 
                fusion_model = torch.compile(fusion_model)
            # if not hasattr(yolo_wrapper, '_compiled_code'):
            #     yolo_wrapper = torch.compile(yolo_wrapper)
        except Exception as e:
            print(f"Compilation failed: {e}")

    # Fusion 모델 동결 (Stage 1에서 학습 완료)
    fusion_model.eval()
    for param in fusion_model.parameters():
        param.requires_grad = False
        
    # YOLO 모델 동결 해제
    yolo_wrapper.unfreeze()
    
    # YOLO 손실 함수 모듈 임포트 및 인스턴스화 (실패 시 에러 발생)
    from utils.loss import ComputeLoss
    
    if not hasattr(yolo_wrapper.model, 'hyp'):
        yolo_wrapper.model.hyp = {'box': 0.05, 'obj': 1.0, 'cls': 0.5, 'anchor_t': 4.0, 'cls_pw': 1.0, 'obj_pw': 1.0, 'fl_gamma': 0.0, 'label_smoothing': 0.0}
    if not hasattr(yolo_wrapper.model, 'gr'):
        yolo_wrapper.model.gr = 1.0
        
    compute_loss = ComputeLoss(yolo_wrapper.model)

    optimizer = optim.AdamW(yolo_wrapper.parameters(), lr=getattr(config, "YOLO_LEARNING_RATE", 1e-4), weight_decay=1e-4)
    accumulation_steps = max(1, int(getattr(config, "ACCUMULATION_STEPS", 1)))
    # OneCycleLR용 total_steps 계산
    total_steps = epochs * ((len(train_dataloader) + accumulation_steps - 1) // accumulation_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=getattr(config, "YOLO_LEARNING_RATE", 5e-4), total_steps=total_steps, pct_start=0.3)
    use_amp = bool(getattr(config, "USE_AMP", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # 📌 체크포인트 저장을 위한 디렉토리 생성 및 베스트 지표 초기화
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    # 📌 [수정] Top-3 체크포인트 저장을 위한 리스트 (loss, path)
    best_yolo_checkpoints = []
    
    for epoch in range(epochs):
        # 📌 [Phase 4] 점진적 동결 해제 (Curriculum Finetuning)
        # 0-3 epoch: Detector Head만 학습하여 사전학습 가중치 보호
        # 4-7 epoch: Neck 레이어까지 해제하여 도메인 최적화
        if epoch == 0:
            yolo_wrapper.unfreeze_head_only()
        elif epoch == 4:
            yolo_wrapper.unfreeze_selective(threshold=27)
            
        # -------------------------
        # 1. Training Phase
        # -------------------------
        yolo_wrapper.train()
        total_train_loss = 0
        
        for batch_idx, (original, saturated, binary, targets) in enumerate(train_dataloader):
            original, saturated, binary, targets = original.to(device), saturated.to(device), binary.to(device), targets.to(device)
            
            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)
            
            autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
            with autocast_ctx:
                # 퓨전 이미지를 생성 (그래디언트 계산 불필요)
                with torch.no_grad():
                    fused = fusion_model(saturated, binary)
                
                # YOLO 예측
                yolo_output = yolo_wrapper(fused)
                
                # YOLOv7 실제 탐지 손실(Standard YOLOv7 detection loss) 계산
                if compute_loss is not None:
                    loss, loss_items = compute_loss(yolo_output, targets)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss = loss / accumulation_steps
                    
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if epoch == 0 and batch_idx == 0:
                has_grad = any(p.grad is not None for p in yolo_wrapper.parameters())
                print(f"[Stage 2 Autograd Check] YOLO model gradients computed: {has_grad}")
                
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_dataloader))
            if is_update_step:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(yolo_wrapper.parameters(), max_norm=1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()  # 📌 배치(업데이트) 단위로 스케줄러 이동
            
            total_train_loss += loss.item() * accumulation_steps
        
        # scheduler.step() # 제거됨
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # -------------------------
        # 2. Validation Phase
        # -------------------------
        yolo_wrapper.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for v_original, v_saturated, v_binary, v_targets in val_dataloader:
                v_original, v_saturated, v_binary, v_targets = v_original.to(device), v_saturated.to(device), v_binary.to(device), v_targets.to(device)
                
                autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
                with autocast_ctx:
                    v_fused = fusion_model(v_saturated, v_binary)
                    v_yolo_output = yolo_wrapper(v_fused, force_train_out=True)
                    
                    if compute_loss is not None:
                        v_loss, _ = compute_loss(v_yolo_output, v_targets)
                    else:
                        v_loss = torch.tensor(0.0, device=device)
                    total_val_loss += v_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Stage 2 Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # -------------------------
        # 3. Top-3 Checkpoint Saving Logic
        # -------------------------
        if len(best_yolo_checkpoints) < 3 or avg_val_loss < best_yolo_checkpoints[-1][0]:
            # 새 체크포인트 저장
            temp_path = os.path.join(config.CHECKPOINT_DIR, f'yolo_model_epoch_{epoch+1}_loss_{avg_val_loss:.4f}.pth')
            torch.save(yolo_wrapper.model.state_dict(), temp_path)
            best_yolo_checkpoints.append((avg_val_loss, temp_path))
            best_yolo_checkpoints.sort(key=lambda x: x[0])
            
            # 상위 3개 초과 시 삭제
            if len(best_yolo_checkpoints) > 3:
                _, old_path = best_yolo_checkpoints.pop(-1)
                if os.path.exists(old_path):
                    os.remove(old_path)
            
            # 순위별 고정 파일명으로 복사
            for i, (loss, path) in enumerate(best_yolo_checkpoints):
                rank_path = os.path.join(config.CHECKPOINT_DIR, f'best_yolo_model_rank{i+1}.pth')
                import shutil
                shutil.copy(path, rank_path)
            
            print(f"  [*] Top-3 YOLO Checkpoints updated! Current Best Val Loss: {best_yolo_checkpoints[0][0]:.4f}")
    
    # 학습이 끝나면 가장 성능이 좋았던 가중치로 복구하여 반환
    if best_yolo_checkpoints:
        best_weights_path = os.path.join(config.CHECKPOINT_DIR, 'best_yolo_model_rank1.pth')
        if os.path.exists(best_weights_path):
            yolo_wrapper.model.load_state_dict(torch.load(best_weights_path))
            print("Loaded Top-1 YOLO checkpoint for final return.")
        
    return yolo_wrapper

if __name__ == "__main__":
    print("Stage 1 & 2 Training script ready.")
