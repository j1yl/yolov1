import os
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision import transforms

class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, S=7, B=2, C=20):
        self.root_dir = root_dir
        self.transform = transform
        self.S = S  # Grid size
        self.B = B  # Number of boxes
        self.C = C  # Number of classes

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')

        if split == 'train':
            split_file = os.path.join(root_dir, 'ImageSets', 'Main', 'train.txt')
        else:
            split_file = os.path.join(root_dir, 'ImageSets', 'Main', 'val.txt')

        with open(split_file, 'r') as f:
            self.filenames = [x.strip() for x in f.readlines()]

        # VOC dataset class names
        self.class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{img_name}.xml")

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, _ = image.shape

        # Parse annotation
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Initialize target tensor
        target = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # Extract bounding boxes
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_idx = self.class_dict[class_name]

            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text) / width
            y_min = float(bbox.find('ymin').text) / height
            x_max = float(bbox.find('xmax').text) / width
            y_max = float(bbox.find('ymax').text) / height

            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Determine which grid cell is responsible for this bbox
            grid_x = int(self.S * x_center)
            grid_y = int(self.S * y_center)

            # Adjust x_center and y_center to be relative to grid cell
            cell_x = self.S * x_center - grid_x
            cell_y = self.S * y_center - grid_y

            # Check if the cell already has an object
            if target[grid_y, grid_x, self.C] == 0:
                # Set class prob and box parameters
                target[grid_y, grid_x, class_idx] = 1
                target[grid_y, grid_x, self.C:self.C+5] = torch.tensor([cell_x, cell_y, box_width, box_height, 1.0])

        if self.transform:
            image = self.transform(image)

        return image, target 