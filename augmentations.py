import random
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLOAugmentations:
    """
    Data augmentation class for YOLO training.
    Implements augmentations similar to those used in the original YOLO paper.
    """
    
    @staticmethod
    def get_train_transforms(img_size=448):
        """
        Get training augmentations.
        
        Args:
            img_size (int): Target image size
            
        Returns:
            albumentations.Compose: Augmentation pipeline
        """
        return A.Compose(
            [
                # Resize to consistent size first
                A.Resize(height=img_size, width=img_size),
                
                # Random scaling and translation (up to 20% of original image size)
                A.Affine(
                    scale=(0.8, 1.2),  # 20% scale up/down
                    translate_percent=(-0.2, 0.2),  # 20% translation
                    rotate=0,  # No rotation as per paper
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                
                # Random exposure and saturation adjustments in HSV color space
                A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=0.5,  # Up to 1.5x saturation (0.5 = 50% increase)
                    val_shift_limit=0.5,  # Up to 1.5x exposure (0.5 = 50% increase)
                    p=0.5
                ),
                
                # Random horizontal flip
                A.HorizontalFlip(p=0.5),
                
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Convert to tensor
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format='albumentations',  # Use albumentations format [x_min, y_min, x_max, y_max]
                label_fields=['class_labels']
            )
        )
    
    @staticmethod
    def get_val_transforms(img_size=448):
        """
        Get validation transforms (only normalization and resizing).
        
        Args:
            img_size (int): Target image size
            
        Returns:
            albumentations.Compose: Transform pipeline
        """
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format='albumentations',  # Use albumentations format [x_min, y_min, x_max, y_max]
                label_fields=['class_labels']
            )
        )
    
    @staticmethod
    def yolo_to_albumentations(bbox, img_width, img_height):
        """
        Convert YOLO format bbox to albumentations format.
        
        Args:
            bbox (list): [x_center, y_center, width, height] in YOLO format
            img_width (int): Image width
            img_height (int): Image height
            
        Returns:
            list: [x_min, y_min, x_max, y_max] in albumentations format
        """
        x_center, y_center, width, height = bbox
        
        # Convert to absolute coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Convert to x_min, y_min, x_max, y_max format
        x_min = x_center - width/2
        y_min = y_center - height/2
        x_max = x_center + width/2
        y_max = y_center + height/2
        
        return [x_min, y_min, x_max, y_max]
    
    @staticmethod
    def albumentations_to_yolo(bbox, img_width, img_height):
        """
        Convert albumentations format bbox to YOLO format.
        
        Args:
            bbox (list): [x_min, y_min, x_max, y_max] in albumentations format
            img_width (int): Image width
            img_height (int): Image height
            
        Returns:
            list: [x_center, y_center, width, height] in YOLO format
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to center coordinates and dimensions
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width/2
        y_center = y_min + height/2
        
        # Convert to relative coordinates
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    @staticmethod
    def convert_yolo_target_to_albumentations(target, S=7, B=2, C=20):
        """
        Convert YOLO target tensor to albumentations format.
        
        Args:
            target (torch.Tensor): YOLO target tensor of shape (S, S, C + 5*B)
            S (int): Grid size
            B (int): Number of boxes per cell
            C (int): Number of classes
            
        Returns:
            list: List of bounding boxes in albumentations format
            list: List of class labels
        """
        boxes = []
        class_labels = []
        
        for i in range(S):
            for j in range(S):
                # Check if there's an object in this cell
                if target[i, j, C] == 1:
                    # Get box coordinates
                    x_center = (j + target[i, j, C]) / S
                    y_center = (i + target[i, j, C+1]) / S
                    width = target[i, j, C+2]
                    height = target[i, j, C+3]
                    
                    # Convert to albumentations format (x_center, y_center, width, height)
                    boxes.append([x_center, y_center, width, height])
                    
                    # Get class label
                    class_idx = torch.argmax(target[i, j, :C]).item()
                    class_labels.append(class_idx)
        
        return boxes, class_labels
    
    @staticmethod
    def convert_albumentations_to_yolo_target(boxes, class_labels, S=7, B=2, C=20):
        """
        Convert albumentations format back to YOLO target tensor.
        
        Args:
            boxes (list): List of bounding boxes in albumentations format
            class_labels (list): List of class labels
            S (int): Grid size
            B (int): Number of boxes per cell
            C (int): Number of classes
            
        Returns:
            torch.Tensor: YOLO target tensor of shape (S, S, C + 5*B)
        """
        target = torch.zeros((S, S, C + 5 * B))
        
        for box, class_idx in zip(boxes, class_labels):
            x_center, y_center, width, height = box
            
            # Determine grid cell
            grid_x = int(S * x_center)
            grid_y = int(S * y_center)
            
            # Adjust coordinates to be relative to grid cell
            cell_x = S * x_center - grid_x
            cell_y = S * y_center - grid_y
            
            # Check if the cell already has an object
            if target[grid_y, grid_x, C] == 0:
                # Set class prob and box parameters
                target[grid_y, grid_x, class_idx] = 1
                target[grid_y, grid_x, C:C+5] = torch.tensor(
                    [cell_x, cell_y, width, height, 1.0]
                )
        
        return target 