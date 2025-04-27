import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VOCDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, S=7, B=2, C=20):
        self.root_dir = root_dir
        self.transform = transform
        self.S = S  # Grid size
        self.B = B  # Number of boxes
        self.C = C  # Number of classes

        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "Annotations")

        if split == "train":
            split_file = os.path.join(root_dir, "ImageSets", "Main", "train.txt")
        else:
            split_file = os.path.join(root_dir, "ImageSets", "Main", "val.txt")

        with open(split_file, "r") as f:
            self.filenames = [x.strip() for x in f.readlines()]

        # VOC dataset class names
        self.class_names = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.class_dict = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{img_name}.xml")

        # Load image using PIL
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)  # Convert to numpy array for albumentations

        # Get image dimensions
        height, width = image.shape[:2]

        # Parse annotation
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Extract bounding boxes and class labels for albumentations
        boxes = []
        class_labels = []
        
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            class_idx = self.class_dict[class_name]

            bbox = obj.find("bndbox")
            x_min = float(bbox.find("xmin").text)
            y_min = float(bbox.find("ymin").text)
            x_max = float(bbox.find("xmax").text)
            y_max = float(bbox.find("ymax").text)
            
            # Convert to normalized coordinates [0, 1] for albumentations
            x_min_norm = x_min / width
            y_min_norm = y_min / height
            x_max_norm = x_max / width
            y_max_norm = y_max / height
            
            # Add to boxes list in albumentations format [x_min, y_min, x_max, y_max]
            boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
            class_labels.append(class_idx)

        # Apply transforms if available
        if self.transform:
            # Albumentations requires named arguments
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]
            
            # Get new image dimensions after transform
            height, width = image.shape[:2]

        # Initialize target tensor
        target = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # Convert boxes to YOLO format and fill target tensor
        for box, class_idx in zip(boxes, class_labels):
            # Boxes are already in normalized format [0, 1]
            x_min, y_min, x_max, y_max = box
            
            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Determine which grid cell is responsible for this bbox
            grid_x = int(self.S * x_center)
            grid_y = int(self.S * y_center)
            
            # Skip if box is outside the grid
            if grid_x >= self.S or grid_y >= self.S:
                continue

            # Adjust x_center and y_center to be relative to grid cell
            cell_x = self.S * x_center - grid_x
            cell_y = self.S * y_center - grid_y

            # Check if the cell already has an object
            if target[grid_y, grid_x, self.C] == 0:
                # Ensure class_idx is an integer
                class_idx_int = int(class_idx)
                
                # Set class prob and box parameters
                target[grid_y, grid_x, class_idx_int] = 1
                target[grid_y, grid_x, self.C : self.C + 5] = torch.tensor(
                    [cell_x, cell_y, box_width, box_height, 1.0]
                )

        return image, target

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes)
        
        if self.transform:
            # Apply basic augmentations
            image = self.transform(image)
            
            # Convert boxes to tensor format [x, y, w, h, class]
            boxes = boxes.clone()
            
            # Scale coordinates to match transformed image size
            boxes[:, 1:] = boxes[:, 1:] * torch.tensor([image.shape[-1], image.shape[-2], 
                                                        image.shape[-1], image.shape[-2]])
        
        # Convert to YOLO format
        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))
        
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            
            # Convert to grid cell coordinates
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            # Ensure we're within bounds
            if i < self.S and j < self.S:
                # Set class one-hot encoding
                label_matrix[i, j, int(class_label)] = 1
                
                # Set box coordinates and confidence
                label_matrix[i, j, self.C:self.C+4] = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, self.C+4] = 1  # confidence
                
                # Set second box to be the same (YOLOv1 only uses one box per cell)
                label_matrix[i, j, self.C+5:self.C+9] = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, self.C+9] = 1  # confidence
        
        return image, label_matrix

def get_transform(train=True):
    if train:
        return A.Compose(
            [
                A.Resize(448, 448),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(448, 448),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
            ),
        )
