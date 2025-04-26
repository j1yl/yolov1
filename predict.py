from datetime import datetime
import os
import torch
import torchvision.transforms as transforms
import cv2

from model import YOLOv1
from utils import convert_cellboxes_to_boxes, non_max_suppression, draw_boxes

def predict(model, image_path, device, class_names):
    """
    Run prediction on an image.

    Args:
        model (YOLOv1): Trained YOLOv1 model
        image_path (str): Path to image file
        device (str): Device to run inference on ('cpu' or 'cuda')
        class_names (list): List of class names
    """
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Original image dimensions for scaling
    orig_height, orig_width = image.shape[:2]

    # Transform image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Post-process predictions
    bboxes, scores, class_ids = convert_cellboxes_to_boxes(predictions)

    # Non-max suppression
    boxes = bboxes[0]  # Take first image in batch
    scores_np = scores[0]
    class_ids_np = class_ids[0]

    if len(boxes) > 0:
        # Apply NMS
        keep_indices = non_max_suppression(boxes, scores_np)

        # Filter boxes
        boxes = [boxes[i] for i in keep_indices]
        scores_np = [scores_np[i] for i in keep_indices]
        class_ids_np = [class_ids_np[i] for i in keep_indices]

        # Draw boxes on image
        draw_boxes(image, boxes, scores_np, class_ids_np, class_names)

        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image with predictions
        output_path = os.path.join("results", f"prediction_{timestamp}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Prediction saved to {output_path}")
    else:
        print("No objects detected.")

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "checkpoints/yolov1_epoch_1.pt"
    IMAGE_PATH = "data/test.jpg"

    # VOC class names
    class_names = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    ]

    # Load model
    model = YOLOv1(in_channels=3, num_boxes=2, num_classes=20).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run prediction
    predict(model, IMAGE_PATH, DEVICE, class_names)

if __name__ == "__main__":
    main() 