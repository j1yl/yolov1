import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Perform non-maximum suppression on boxes.

    Args:
        boxes (list): List of boxes [x1, y1, x2, y2]
        scores (list): List of confidence scores
        iou_threshold (float): Threshold for IoU

    Returns:
        list: Indices of boxes to keep
    """
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort by confidence score
    sorted_indices = np.argsort(scores)[::-1]

    keep_indices = []
    while sorted_indices.size > 0:
        # Pick the box with highest confidence
        best_idx = sorted_indices[0]
        keep_indices.append(best_idx)

        if len(sorted_indices) == 1:
            break

        # Calculate IoU with the best box
        best_box = boxes[best_idx]
        other_boxes = boxes[sorted_indices[1:]]

        ious = calculate_iou_np(best_box, other_boxes)

        # Keep boxes with IoU less than threshold
        mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    return keep_indices


def calculate_iou_np(box, boxes):
    """
    Calculate IoU between a box and a list of boxes.

    Args:
        box (numpy.ndarray): Single box [x1, y1, x2, y2]
        boxes (numpy.ndarray): Array of boxes (..., 4)

    Returns:
        numpy.ndarray: IoU values
    """
    # Box coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection + 1e-6

    # IoU
    iou = intersection / union

    return iou


def convert_cellboxes_to_boxes(predictions, S=7, B=2, C=20):
    """
    Convert YOLO predictions to bounding boxes.

    Args:
        predictions (tensor): Model predictions of shape (batch_size, S, S, C + B*5)
        S (int): Grid size
        B (int): Number of boxes per grid cell
        C (int): Number of classes

    Returns:
        tuple: (all_boxes, all_scores, all_class_ids)
    """
    batch_size = predictions.shape[0]
    predictions = predictions.to("cpu")

    bboxes = []
    class_scores = []
    class_ids = []

    # Process each example in batch
    for batch_idx in range(batch_size):
        boxes = []
        scores = []
        class_preds = []

        # Process each grid cell
        for i in range(S):
            for j in range(S):
                # Get class probabilities and box scores
                class_pred = predictions[batch_idx, i, j, :C]
                class_id = torch.argmax(class_pred)
                class_score = class_pred[class_id]

                # Get box coordinates and confidence for each predicted box
                for b in range(B):
                    box_start_idx = C + b * 5
                    box_pred = predictions[
                        batch_idx, i, j, box_start_idx : box_start_idx + 5
                    ]
                    confidence = box_pred[4]

                    # Only process box if confidence is above threshold
                    if confidence > 0.5:
                        # Convert to absolute coordinates
                        x_center = (box_pred[0] + j) / S
                        y_center = (box_pred[1] + i) / S
                        width = box_pred[2]
                        height = box_pred[3]

                        # Convert to corner format (x1, y1, x2, y2)
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2

                        boxes.append([x1, y1, x2, y2])
                        scores.append(confidence * class_score)
                        class_preds.append(class_id)

        bboxes.append(boxes)
        class_scores.append(scores)
        class_ids.append(class_preds)

    return bboxes, class_scores, class_ids


def draw_boxes(image, boxes, scores, class_ids, class_names):
    """
    Draw bounding boxes on image.

    Args:
        image (numpy.ndarray): Image
        boxes (list): List of bounding boxes [x1, y1, x2, y2]
        scores (list): List of confidence scores
        class_ids (list): List of class indices
        class_names (list): List of class names
    """
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Assign colors for each class
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]

    # Draw boxes
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box

        # Convert to pixel coordinates
        height, width, _ = image.shape
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        # Create rectangle patch
        color = colors[int(class_id)]
        box_width = x2 - x1
        box_height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1),
            box_width,
            box_height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )

        # Add rectangle to plot
        ax.add_patch(rect)

        # Add label
        class_name = class_names[int(class_id)]
        plt.text(
            x1,
            y1,
            f"{class_name} {score:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor=color, alpha=0.5),
        )

    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()


def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP) for object detection.

    Args:
        predictions: Tuple of (bboxes, scores, class_ids) lists for each image
        targets: Tensor of shape (batch_size, S, S, C + B*5) containing target boxes
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        float: Mean Average Precision across all classes
    """
    num_classes = 20  # PASCAL VOC has 20 classes
    aps = []

    # Unpack predictions
    all_bboxes, all_scores, all_class_ids = predictions

    # Process each class
    for class_id in range(num_classes):
        all_predictions = []
        all_targets = []

        # Collect predictions and targets for this class
        for img_idx in range(len(all_bboxes)):
            # Get predictions for this class
            class_mask = [cid == class_id for cid in all_class_ids[img_idx]]
            if any(class_mask):
                class_bboxes = [
                    box for i, box in enumerate(all_bboxes[img_idx]) if class_mask[i]
                ]
                class_scores = [
                    score
                    for i, score in enumerate(all_scores[img_idx])
                    if class_mask[i]
                ]
                all_predictions.extend(list(zip(class_bboxes, class_scores)))

            # Get targets for this class from the 7x7 grid
            target = targets[img_idx]
            S = target.shape[0]  # Grid size
            C = 20  # Number of classes
            B = 2  # Number of boxes per cell

            for i in range(S):
                for j in range(S):
                    # Check each box in the cell
                    for b in range(B):
                        box_start_idx = C + b * 5
                        box_data = target[i, j, box_start_idx : box_start_idx + 5]
                        confidence = box_data[4]

                        # If this box contains an object
                        if confidence > 0.5:
                            # Get class probabilities for this cell
                            class_probs = target[i, j, :C]
                            target_class = torch.argmax(class_probs).item()

                            # If this is the class we're looking for
                            if target_class == class_id:
                                # Convert YOLO format to [x1, y1, x2, y2]
                                x, y, w, h = box_data[:4]
                                x_center = (x + j) / S
                                y_center = (y + i) / S
                                x1 = x_center - w / 2
                                y1 = y_center - h / 2
                                x2 = x_center + w / 2
                                y2 = y_center + h / 2
                                all_targets.append([x1, y1, x2, y2])

        if all_predictions and all_targets:
            ap = calculate_ap(all_predictions, all_targets, iou_threshold)
            aps.append(ap)

    return np.mean(aps) if aps else 0.0


def calculate_ap(predictions, targets, iou_threshold=0.5):
    """
    Calculate Average Precision for a single class.

    Args:
        predictions: List of (bbox, score) tuples
        targets: List of target bounding boxes
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        float: Average Precision
    """
    # Sort predictions by confidence
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    used_targets = set()

    # Process each prediction
    for pred_idx, (pred_box, score) in enumerate(predictions):
        best_iou = -np.inf
        best_target_idx = -1

        # Find best matching target
        for target_idx, target_box in enumerate(targets):
            if target_idx in used_targets:
                continue

            iou = calculate_iou_np(pred_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_target_idx = target_idx

        # Update TP/FP based on IoU threshold
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            used_targets.add(best_target_idx)
        else:
            fp[pred_idx] = 1

    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(targets)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap = ap + p / 11.0

    return ap
