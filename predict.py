from datetime import datetime
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

from model import YOLOv1
from utils import convert_cellboxes_to_boxes, non_max_suppression

# VOC class names
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def load_model(checkpoint_path, device):
    """
    Load a trained YOLOv1 model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded model
    """
    # Initialize model
    model = YOLOv1(in_channels=3, num_boxes=2, num_classes=20).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # If the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    return model

def preprocess_image(image_path, img_size=448):
    """
    Preprocess image for model input.

    Args:
        image_path: Path to the image file
        img_size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor and original image dimensions
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Get original dimensions
    orig_width, orig_height = image.size
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return img_tensor, (orig_width, orig_height)

def convert_cellboxes_to_boxes(predictions, S=7, B=2, C=20):
    """
    Convert YOLO predictions to bounding boxes.
    
    Args:
        predictions: Model predictions [batch_size, S, S, B*5 + C]
        S: Grid size
        B: Number of boxes per cell
        C: Number of classes
        
    Returns:
        List of bounding boxes, scores, and class IDs
    """
    batch_size = predictions.shape[0]
    
    # Reshape predictions
    predictions = predictions.reshape(batch_size, S, S, B*5 + C)
    
    # Extract confidence scores
    confidences = predictions[..., C::5]
    
    # Find best box for each cell
    best_box_idx = torch.argmax(confidences, dim=-1)
    
    # Initialize lists for boxes, scores, and class IDs
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    # Process each image in the batch
    for b in range(batch_size):
        boxes = []
        scores = []
        class_ids = []
        
        # Process each cell
        for i in range(S):
            for j in range(S):
                # Get the best box index for this cell
                box_idx = best_box_idx[b, i, j].item()
                
                # Get confidence score
                confidence = predictions[b, i, j, C + box_idx*5 + 4].item()
                
                # Skip if confidence is too low
                if confidence < 0.5:
                    continue
                
                # Get box coordinates
                x = (j + predictions[b, i, j, C + box_idx*5 + 0].item()) / S
                y = (i + predictions[b, i, j, C + box_idx*5 + 1].item()) / S
                w = predictions[b, i, j, C + box_idx*5 + 2].item()
                h = predictions[b, i, j, C + box_idx*5 + 3].item()
                
                # Convert to [x1, y1, x2, y2] format
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                
                # Get class probabilities
                class_probs = predictions[b, i, j, :C].cpu().numpy()
                class_id = np.argmax(class_probs)
                class_score = class_probs[class_id]
                
                # Only add if class score is high enough
                if class_score > 0.5:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence * class_score)
                    class_ids.append(class_id)
        
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_class_ids.append(class_ids)
    
    return all_boxes, all_scores, all_class_ids

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Apply non-max suppression to avoid detecting multiple bounding boxes for the same object.
    
    Args:
        boxes: List of bounding boxes in [x1, y1, x2, y2] format
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Indices of boxes to keep
    """
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []
    
    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate area
    area = (x2 - x1) * (y2 - y1)
    
    # Get indices of boxes sorted by score
    indices = scores.argsort()[::-1]
    
    keep = []
    
    while len(indices) > 0:
        # Get the index of the box with highest score
        i = indices[0]
        keep.append(i)
        
        # If this is the last box, break
        if len(indices) == 1:
            break
        
        # Get coordinates of intersection
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        # Calculate width and height of intersection
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Calculate intersection area
        intersection = w * h
        
        # Calculate union area
        union = area[i] + area[indices[1:]] - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        # Get indices of boxes with IoU <= threshold
        indices = indices[1:][iou <= iou_threshold]
    
    return keep

def visualize_predictions(image_path, boxes, scores, class_ids, class_names, output_path=None):
    """
    Visualize predictions on the image.
    
    Args:
        image_path: Path to the original image
        boxes: List of bounding boxes in [x1, y1, x2, y2] format
        scores: List of confidence scores
        class_ids: List of class IDs
        class_names: List of class names
        output_path: Path to save the visualization (optional)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # Define colors for different classes
    colors = plt.cm.hsv(np.linspace(0, 1, len(class_names)))
    
    # Draw boxes
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        # Scale box coordinates to image size
        x1, y1, x2, y2 = box
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        
        # Get color for this class
        color = colors[class_id]
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add class label and score
        class_name = class_names[class_id]
        label = f"{class_name}: {score:.2f}"
        ax.text(
            x1, y1 - 5, label,
            color='white', fontsize=12, fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.5, edgecolor='none', pad=0)
        )
    
    # Remove axes
    ax.axis('off')
    
    # Generate output path if not provided
    if output_path is None:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get original filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create output path
        output_path = os.path.join("results", f"{base_name}_{timestamp}.jpg")
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"Saved visualization to {output_path}")
    
    # Close the figure to free memory
    plt.close()

def predict(model, image_path, device, class_names, output_path=None, conf_threshold=0.5):
    """
    Run prediction on an image.
    
    Args:
        model: Trained YOLOv1 model
        image_path: Path to the image file
        device: Device to run inference on
        class_names: List of class names
        output_path: Path to save the visualization (optional)
        conf_threshold: Confidence threshold for predictions
    """
    # Preprocess image
    img_tensor, (orig_width, orig_height) = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Convert predictions to boxes
    boxes, scores, class_ids = convert_cellboxes_to_boxes(predictions)
    
    # Apply non-max suppression
    if len(boxes[0]) > 0:
        keep_indices = non_max_suppression(boxes[0], scores[0])
        boxes = [boxes[0][i] for i in keep_indices]
        scores = [scores[0][i] for i in keep_indices]
        class_ids = [class_ids[0][i] for i in keep_indices]
    
    # Filter by confidence threshold
    filtered_boxes = []
    filtered_scores = []
    filtered_class_ids = []
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score >= conf_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filtered_class_ids.append(class_id)
    
    # Visualize predictions
    visualize_predictions(
        image_path, filtered_boxes, filtered_scores, filtered_class_ids,
        class_names, output_path
    )
    
    return filtered_boxes, filtered_scores, filtered_class_ids

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="YOLOv1 Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, default="checkpoints/yolov1_epoch_15.pt", help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to save the visualization (optional)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for predictions")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, device)

    # Run prediction
    predict(
        model, args.image, device, VOC_CLASSES,
        args.output, args.conf
    )

if __name__ == "__main__":
    main() 