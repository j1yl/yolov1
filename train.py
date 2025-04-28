import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import json
from datetime import datetime
import csv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import YOLOv1
from dataset import VOCDataset
from loss import YOLOLoss
from utils import convert_cellboxes_to_boxes

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, learning_rate, log_loss_csv=None):
    model.train()
    loop = tqdm.tqdm(train_loader, leave=True)
    mean_loss = []
    
    # Detailed loss components
    coord_losses = []
    obj_losses = []
    noobj_losses = []
    class_losses = []
    
    epoch_start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(loop):
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        predictions = model(images)
        loss, loss_components = loss_fn(predictions, targets)
        mean_loss.append(loss.item())
        
        # Collect detailed loss components
        coord_losses.append(loss_components['coord_loss'].item())
        obj_losses.append(loss_components['obj_loss'].item())
        noobj_losses.append(loss_components['noobj_loss'].item())
        class_losses.append(loss_components['class_loss'].item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), lr=learning_rate)
    
    epoch_time = time.time() - epoch_start_time
    epoch_mean_loss = sum(mean_loss) / len(mean_loss)
    
    # Calculate mean of each loss component
    stats = {
        'epoch': epoch,
        'mean_loss': epoch_mean_loss,
        'coord_loss': sum(coord_losses) / len(coord_losses),
        'obj_loss': sum(obj_losses) / len(obj_losses),
        'noobj_loss': sum(noobj_losses) / len(noobj_losses),
        'class_loss': sum(class_losses) / len(class_losses),
        'time': epoch_time,
        'learning_rate': learning_rate
    }
    
    # Log loss components to CSV if path provided
    if log_loss_csv is not None:
        file_exists = os.path.isfile(log_loss_csv)
        with open(log_loss_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(stats.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)
    
    print(f"Epoch {epoch} mean loss: {epoch_mean_loss:.4f}, Time: {epoch_time:.2f}s, LR: {learning_rate:.6f}")
    print(f"  coord_loss: {stats['coord_loss']:.4f}, obj_loss: {stats['obj_loss']:.4f}, noobj_loss: {stats['noobj_loss']:.4f}, class_loss: {stats['class_loss']:.4f}")
    return stats

def validate(model, val_loader, device, epoch):
    """
    Validate the model on the validation dataset using torchmetrics' MeanAveragePrecision.
    """
    model.eval()
    metric = MeanAveragePrecision(
        iou_type="bbox",
        box_format="xyxy",
        class_metrics=True,
        iou_thresholds=[0.1, 0.25, 0.5, 0.75]
    )
    num_batches = len(val_loader)
    print(f"\nValidating epoch {epoch}...")
    
    total_predictions = 0
    total_targets = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            predictions = model(images)
            
            # Convert predictions to boxes with lower confidence threshold
            all_boxes, all_scores, all_class_ids = convert_cellboxes_to_boxes(predictions, conf_threshold=0.1)
            
            batch_size = images.shape[0]
            preds = []
            gts = []
            
            for i in range(batch_size):
                # Format predictions for torchmetrics
                boxes = torch.tensor(all_boxes[i], dtype=torch.float32, device=device) if len(all_boxes[i]) > 0 else torch.zeros((0, 4), dtype=torch.float32, device=device)
                scores = torch.tensor(all_scores[i], dtype=torch.float32, device=device) if len(all_scores[i]) > 0 else torch.zeros((0,), dtype=torch.float32, device=device)
                labels = torch.tensor(all_class_ids[i], dtype=torch.int64, device=device) if len(all_class_ids[i]) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
                
                total_predictions += len(boxes)
                preds.append({"boxes": boxes, "scores": scores, "labels": labels})
                
                # Format ground truth for torchmetrics
                gt_boxes = []
                gt_labels = []
                S = targets.shape[1]
                C = 20
                B = 2
                target = targets[i].cpu() if targets.is_cuda else targets[i]
                
                for y in range(S):
                    for x in range(S):
                        for b in range(B):
                            box_start = C + b * 5
                            box_data = target[y, x, box_start:box_start+5]
                            if box_data[4] > 0.5:
                                class_probs = target[y, x, :C]
                                class_id = torch.argmax(class_probs).item()
                                xc = (x + box_data[0].item()) / S
                                yc = (y + box_data[1].item()) / S
                                w = box_data[2].item()
                                h = box_data[3].item()
                                x1 = xc - w / 2
                                y1 = yc - h / 2
                                x2 = xc + w / 2
                                y2 = yc + h / 2
                                
                                # Ensure coordinates are within [0, 1]
                                x1 = max(0, min(1, x1))
                                y1 = max(0, min(1, y1))
                                x2 = max(0, min(1, x2))
                                y2 = max(0, min(1, y2))
                                
                                gt_boxes.append([x1, y1, x2, y2])
                                gt_labels.append(class_id)
                
                total_targets += len(gt_boxes)
                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32, device=device) if len(gt_boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32, device=device)
                gt_labels = torch.tensor(gt_labels, dtype=torch.int64, device=device) if len(gt_labels) > 0 else torch.zeros((0,), dtype=torch.int64, device=device)
                gts.append({"boxes": gt_boxes, "labels": gt_labels})
            
            metric.update(preds, gts)
            if batch_idx % 10 == 0:
                print(f"Validation progress: {batch_idx}/{num_batches} batches")
    
    results = metric.compute()
    
    # Print detailed metrics
    print(f"\nValidation Metrics for Epoch {epoch}:")
    print(f"Total predictions: {total_predictions}")
    print(f"Total ground truth boxes: {total_targets}")
    print(f"mAP@0.5: {results['map_50'].item():.4f}")
    print(f"mAP@0.75: {results['map_75'].item():.4f}")
    print(f"mAP: {results['map'].item():.4f}")
    
    # Print per-class AP
    print("\nPer-class AP@0.5:")
    if 'map_50_per_class' in results and results['map_50_per_class'] is not None:
        for i, ap in enumerate(results['map_50_per_class']):
            print(f"Class {i}: {ap.item():.4f}")
    else:
        print("Per-class AP@0.5 not available (no correct predictions or metric not computed).")
    
    return results['map'].item()

def get_learning_rate(epoch, warmup_epochs=5, total_epochs=135):
    """
    Implement a more conservative learning rate schedule:
    - First warmup_epochs: gradually increase from 10^-4 to 10^-3
    - Next 75 epochs: 10^-3
    - Next 30 epochs: 10^-4
    - Final 30 epochs: 10^-5
    """
    if epoch < warmup_epochs:
        # Linear warmup from 10^-4 to 10^-3
        return 1e-4 + (1e-3 - 1e-4) * (epoch / warmup_epochs)
    elif epoch < warmup_epochs + 75:
        # 10^-3 for 75 epochs
        return 1e-3
    elif epoch < warmup_epochs + 75 + 30:
        # 10^-4 for 30 epochs
        return 1e-4
    else:
        # 10^-5 for the remaining epochs
        return 1e-5

def main():
    INITIAL_LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 84
    WEIGHT_DECAY = 0.0005
    EPOCHS = 135
    WARMUP_EPOCHS = 5
    NUM_WORKERS = 8
    PIN_MEMORY = True

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.json"
    loss_log_csv = f"logs/loss_components_{timestamp}.csv"
    
    # Load datasets with no augmentations
    train_dataset = VOCDataset(
        "data/VOC2012",
        split="train",
        transform=None,
        S=7,
        B=2,
        C=20
    )
    
    val_dataset = VOCDataset(
        "data/VOC2012",
        split="val",
        transform=None,
        S=7,
        B=2,
        C=20
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    
    model = YOLOv1(in_channels=3, num_boxes=2, num_classes=20, dropout_rate=0.5).to(DEVICE)
    
    optimizer = optim.Adam(
        model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YOLOLoss()

    training_stats = {
        'config': {
            'initial_learning_rate': INITIAL_LR,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'warmup_epochs': WARMUP_EPOCHS,
            'device': DEVICE,
            'model': 'YOLOv1',
            'dataset': 'PASCAL VOC 2012',
            'dropout_rate': 0.5,
            'lambda_coord': loss_fn.lambda_coord,
            'lambda_obj': loss_fn.lambda_obj,
            'lambda_noobj': loss_fn.lambda_noobj,
            'lambda_class': loss_fn.lambda_class,
        },
        'epochs': []
    }
    
    start_time = time.time()
    best_map = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        current_lr = get_learning_rate(epoch, WARMUP_EPOCHS, EPOCHS)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        epoch_stats = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch, current_lr, log_loss_csv=loss_log_csv)
        
        # Validation phase (every 15 epochs)
        if (epoch + 1) % 15 == 0 or epoch == 0:
            val_map = validate(model, val_loader, DEVICE, epoch + 1)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_stats['mean_loss'],
                'val_map': val_map,
                'learning_rate': current_lr
            }
            torch.save(checkpoint, os.path.join("checkpoints", f"yolov1_epoch_{epoch+1}.pt"))
            
            # Log statistics
            stats = {
                'epoch': epoch + 1,
                'train_loss': epoch_stats['mean_loss'],
                'val_map': val_map,
                'learning_rate': current_lr
            }
            with open(log_file, 'a') as f:
                json.dump(stats, f)
                f.write('\n')
            
            # Save best model
            if val_map > best_map:
                best_map = val_map
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "map": val_map,
                    "learning_rate": current_lr
                }, os.path.join("checkpoints", "best_model.pth.tar"))
                print(f"New best model saved with mAP: {val_map:.4f}")
        
        # Save checkpoint every 15 epochs
        if (epoch + 1) % 15 == 0 or epoch == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_stats['mean_loss'],
                'learning_rate': current_lr
            }
            torch.save(checkpoint, os.path.join("checkpoints", f"yolov1_epoch_{epoch+1}.pt"))
        
        # Add epoch stats to training stats
        training_stats['epochs'].append(epoch_stats)
        
        # Save training stats to file
        with open(log_file, 'w') as f:
            json.dump(training_stats, f, indent=4)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    
    training_stats['total_time'] = total_time
    training_stats['best_map'] = best_map

    with open(log_file, 'w') as f:
        json.dump(training_stats, f, indent=4)
    
    print(f"Training statistics saved to {log_file}")

if __name__ == "__main__":
    main() 