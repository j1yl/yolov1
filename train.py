import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
import json
from datetime import datetime
import math

from model import YOLOv1
from dataset import VOCDataset
from loss import YOLOLoss
from utils import convert_cellboxes_to_boxes, calculate_map
from augmentations import YOLOAugmentations

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, learning_rate):
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
    
    print(f"Epoch {epoch} mean loss: {epoch_mean_loss:.4f}, Time: {epoch_time:.2f}s, LR: {learning_rate:.6f}")
    return stats

def validate(model, val_loader, device, epoch):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    total_map = 0
    num_batches = len(val_loader)
    
    print(f"\nValidating epoch {epoch}...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Validation progress: {batch_idx}/{num_batches} batches")
                
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            predictions = convert_cellboxes_to_boxes(predictions)
            
            # Calculate mAP for this batch
            batch_map = calculate_map(predictions, targets)
            total_map += batch_map
            
    avg_map = total_map / num_batches
    print(f"Epoch {epoch} Validation mAP: {avg_map:.4f}")
    return avg_map

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
    # Configuration
    INITIAL_LR = 1e-3  # Starting learning rate
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 80  # RTX 3090 has 24GB VRAM, can handle larger batches
    WEIGHT_DECAY = 0
    EPOCHS = 135  # same as original paper
    WARMUP_EPOCHS = 5  # Number of epochs to warm up learning rate
    NUM_WORKERS = 4
    PIN_MEMORY = True
    # LOAD_PRETRAINED = False
    # LOAD_PRETRAINED_MODEL_PATH = ""
    # IMG_DIR = "data/VOC2012/JPEGImages"
    # LABEL_DIR = "data/VOC2012/Annotations"
    
    # Create directories for logs and checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.json"
    
    # Get augmentations from YOLOAugmentations class
    train_transform = YOLOAugmentations.get_train_transforms(img_size=448)
    val_transform = YOLOAugmentations.get_val_transforms(img_size=448)
    
    # Load datasets with the new augmentations
    train_dataset = VOCDataset(
        "data/VOC2012",
        split="train",
        transform=train_transform,
        S=7,
        B=2,
        C=20
    )
    
    val_dataset = VOCDataset(
        "data/VOC2012",
        split="val",
        transform=val_transform,
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
    
    # Initialize model with dropout rate 0.5 after the first connected layer
    model = YOLOv1(in_channels=3, num_boxes=2, num_classes=20, dropout_rate=0.5).to(DEVICE)
    
    # Initialize optimizer with initial learning rate
    optimizer = optim.Adam(
        model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YOLOLoss()
    
    # Load pretrained model if specified
    # if LOAD_PRETRAINED:
    #     checkpoint = torch.load(LOAD_PRETRAINED_MODEL_PATH)
    #     model.load_state_dict(checkpoint["state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    
    # Training statistics
    training_stats = {
        'config': {
            'initial_learning_rate': INITIAL_LR,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'warmup_epochs': WARMUP_EPOCHS,
            'device': DEVICE,
            'model': 'YOLOv1',
            'dataset': 'PASCAL VOC 2012',
            'dropout_rate': 0.5
        },
        'epochs': []
    }
    
    # Training loop
    start_time = time.time()
    best_map = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Get learning rate for this epoch
        current_lr = get_learning_rate(epoch, WARMUP_EPOCHS, EPOCHS)
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Training phase
        epoch_stats = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE, epoch, current_lr)
        
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
    
    # Save final training stats
    training_stats['total_time'] = total_time
    training_stats['best_map'] = best_map
    with open(log_file, 'w') as f:
        json.dump(training_stats, f, indent=4)
    
    print(f"Training statistics saved to {log_file}")

if __name__ == "__main__":
    main() 