# TODO: implement evaluation metrics i.e. mAP and compare against the original YOLO v1 paper results
# https://arxiv.org/pdf/1506.02640

# Run the training script (python train.py)
# Implement evaluation metrics in a new file (e.g., evaluate.py)
# Run evaluation on the validation set
# Collect all statistics (training loss, mAP, training time)
# Create visualizations (loss curves, mAP values)
# Write your 1-page report with all the required information

import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from model import YOLOv1
from dataset import VOCDataset
from utils import convert_cellboxes_to_boxes, calculate_map

def evaluate_checkpoint(model, val_loader, device, checkpoint_path):
    """
    Evaluate a single checkpoint on the validation set.
    
    Args:
        model (YOLOv1): YOLOv1 model
        val_loader (DataLoader): Validation data loader
        device (str): Device to run evaluation on
        checkpoint_path (str): Path to checkpoint file
    
    Returns:
        float: mAP score
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    total_map = 0
    num_batches = len(val_loader)
    
    print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            predictions = convert_cellboxes_to_boxes(predictions)
            
            # Calculate mAP for this batch
            batch_map = calculate_map(predictions, targets)
            total_map += batch_map
            
    avg_map = total_map / num_batches
    print(f"mAP: {avg_map:.4f}")
    return avg_map

def plot_map_curve(map_scores, epochs, output_dir):
    """
    Plot mAP curve over epochs.
    
    Args:
        map_scores (list): List of mAP scores
        epochs (list): List of epochs
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, map_scores, 'b-', marker='o')
    plt.title('mAP over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'map_curve.png'))
    plt.close()

def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 80
    NUM_WORKERS = 8
    PIN_MEMORY = True
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"logs/evaluation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])
    
    # Load validation dataset
    val_dataset = VOCDataset(
        "data/VOC2012",
        split="val",
        transform=transform,
        S=7,
        B=2,
        C=20
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    
    # Initialize model
    model = YOLOv1(in_channels=3, num_boxes=2, num_classes=20).to(DEVICE)
    
    # Get all checkpoint files
    checkpoint_dir = "checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Evaluate each checkpoint
    results = []
    map_scores = []
    epochs = []
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        epoch = int(checkpoint_file.split('_')[-1].split('.')[0])
        
        map_score = evaluate_checkpoint(model, val_loader, DEVICE, checkpoint_path)
        
        results.append({
            'checkpoint': checkpoint_file,
            'epoch': epoch,
            'map': map_score
        })
        
        map_scores.append(map_score)
        epochs.append(epoch)
    
    # Plot mAP curve
    plot_map_curve(map_scores, epochs, output_dir)
    
    # Save results
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Results saved to: {output_dir}")
    print(f"Number of checkpoints evaluated: {len(results)}")
    print(f"Best mAP: {max(map_scores):.4f} at epoch {epochs[map_scores.index(max(map_scores))]}")
    print(f"Average mAP: {sum(map_scores)/len(map_scores):.4f}")

if __name__ == "__main__":
    main()