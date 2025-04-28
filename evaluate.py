import os
import json
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from model import YOLOv1
from dataset import VOCDataset
from utils import convert_cellboxes_to_boxes, calculate_map


def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (YOLOv1): Trained YOLOv1 model
        data_loader (DataLoader): Validation data loader
        device (str): Device to run evaluation on ('cpu' or 'cuda')
        iou_threshold (float): IoU threshold for mAP calculation

    Returns:
        float: Mean Average Precision (mAP)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(images)

            # Convert predictions to bounding boxes
            bboxes, scores, class_ids = convert_cellboxes_to_boxes(predictions)

            # Store predictions and targets
            all_predictions.append((bboxes, scores, class_ids))
            all_targets.append(targets)

    # Calculate mAP
    mAP = calculate_map(all_predictions, all_targets, iou_threshold)

    return mAP


def evaluate_checkpoints(
    checkpoints_dir, val_dataset, device, batch_size=16, iou_threshold=0.5
):
    """
    Evaluate multiple model checkpoints and save results.

    Args:
        checkpoints_dir (str): Directory containing model checkpoints
        val_dataset (VOCDataset): Validation dataset
        device (str): Device to run evaluation on ('cpu' or 'cuda')
        batch_size (int): Batch size for evaluation
        iou_threshold (float): IoU threshold for mAP calculation

    Returns:
        dict: Dictionary with evaluation results
    """
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Get checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Initialize results
    results = []

    # Evaluate each checkpoint
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
        epoch = int(checkpoint_file.split("_")[-1].split(".")[0])

        print(f"Evaluating checkpoint: {checkpoint_file}")

        # Load model
        model = YOLOv1(in_channels=3, num_boxes=2, num_classes=20).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluate model
        mAP = evaluate_model(model, val_loader, device, iou_threshold)

        # Store results
        results.append({"checkpoint": checkpoint_file, "epoch": epoch, "map": mAP})

        print(f"Epoch {epoch}, mAP: {mAP:.6f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"evaluation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Plot mAP curve
    epochs = [r["epoch"] for r in results]
    maps = [r["map"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, maps, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision (mAP) vs. Epoch")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "map_curve.png"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv1 model")
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/VOCdevkit/VOC2012",
        help="Path to VOC dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for mAP calculation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    args = parser.parse_args()

    # Create validation dataset
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]
    )

    val_dataset = VOCDataset(
        root_dir=args.data_dir, split="val", transform=transform, S=7, B=2, C=20
    )

    # Evaluate checkpoints
    results = evaluate_checkpoints(
        args.checkpoints_dir,
        val_dataset,
        args.device,
        args.batch_size,
        args.iou_threshold,
    )

    # Print summary
    best_map = max(results, key=lambda x: x["map"])
    print(f"\nEvaluation complete!")
    print(f"Best mAP: {best_map['map']:.6f} at epoch {best_map['epoch']}")
    print(f"Results saved to: evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


if __name__ == "__main__":
    main()
