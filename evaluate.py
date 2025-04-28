import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model import YOLOv1
from dataset import VOCDataset, get_transform
from utils import convert_cellboxes_to_boxes

def evaluate_checkpoint(model, val_loader, device, checkpoint_path):
    """
    Evaluate a single checkpoint on the validation set using torchvision's mAP implementation.
    
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
    
    # Initialize metric
    metric = MeanAveragePrecision()
    
    print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions
            predictions = model(images)
            pred_boxes, pred_scores, pred_labels = convert_cellboxes_to_boxes(predictions)
            
            # Convert predictions to torchvision format
            batch_preds = []
            for boxes, scores, labels in zip(pred_boxes, pred_scores, pred_labels):
                if len(boxes) > 0:
                    # Convert boxes to tensor format [x1, y1, x2, y2]
                    boxes_tensor = torch.tensor(boxes, device=device)
                    scores_tensor = torch.tensor(scores, device=device)
                    labels_tensor = torch.tensor(labels, device=device)
                    
                    batch_preds.append({
                        'boxes': boxes_tensor,
                        'scores': scores_tensor,
                        'labels': labels_tensor
                    })
                else:
                    # Empty prediction
                    batch_preds.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'scores': torch.zeros(0, device=device),
                        'labels': torch.zeros(0, dtype=torch.int64, device=device)
                    })
            
            # Convert targets to torchvision format
            batch_targets = []
            for target in targets:
                # Extract ground truth boxes and labels from YOLO format
                S = target.shape[0]  # Grid size
                C = 20  # Number of classes
                B = 2  # Number of boxes per cell
                
                gt_boxes = []
                gt_labels = []
                
                for i in range(S):
                    for j in range(S):
                        # Check each box in the cell
                        for b in range(B):
                            box_start_idx = C + b * 5
                            box_data = target[i, j, box_start_idx:box_start_idx + 5]
                            confidence = box_data[4]
                            
                            if confidence > 0.5:
                                # Get class probabilities
                                class_probs = target[i, j, :C]
                                label = torch.argmax(class_probs).item()
                                
                                # Convert YOLO format to [x1, y1, x2, y2]
                                x, y, w, h = box_data[:4]
                                x_center = (x + j) / S
                                y_center = (y + i) / S
                                x1 = x_center - w / 2
                                y1 = y_center - h / 2
                                x2 = x_center + w / 2
                                y2 = y_center + h / 2
                                
                                gt_boxes.append([x1, y1, x2, y2])
                                gt_labels.append(label)
                
                if len(gt_boxes) > 0:
                    batch_targets.append({
                        'boxes': torch.tensor(gt_boxes, device=device),
                        'labels': torch.tensor(gt_labels, device=device)
                    })
                else:
                    batch_targets.append({
                        'boxes': torch.zeros((0, 4), device=device),
                        'labels': torch.zeros(0, dtype=torch.int64, device=device)
                    })
            
            # Update metric
            metric.update(batch_preds, batch_targets)
    
    # Compute final metrics
    results = metric.compute()
    map_score = results['map'].item()
    print(f"mAP: {map_score:.4f}")
    return map_score

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
    
    # Get validation transform
    transform = get_transform(train=False)
    
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