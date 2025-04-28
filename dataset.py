import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

        # Load image using PIL and resize to 448x448
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        image = image.resize((448, 448), Image.BILINEAR)
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Get image dimensions (after resize)
        height, width = 448, 448

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
            
            # Convert to normalized coordinates [0, 1]
            x_min_norm = x_min / orig_width
            y_min_norm = y_min / orig_height
            x_max_norm = x_max / orig_width
            y_max_norm = y_max / orig_height
            
            # Add to boxes list in [x_min, y_min, x_max, y_max]
            boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
            class_labels.append(class_idx)

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
                # Set one-hot class vector
                target[grid_y, grid_x, :self.C] = 0
                target[grid_y, grid_x, int(class_idx)] = 1.0
                # Set box parameters and objectness for ALL B boxes (so loss can match best box)
                for b in range(self.B):
                    start = self.C + b * 5
                    target[grid_y, grid_x, start:start+5] = torch.tensor(
                        [cell_x, cell_y, box_width, box_height, 1.0]
                    )
        target = torch.clamp(target, 0, 1)
        return image, target


def visualize_voc_batch(dataset, batch_size=4, class_names=None, output_dir="visualize_gt"):
    """
    Visualize a batch of images from VOCDataset with ground truth boxes and class labels.
    Args:
        dataset: VOCDataset instance
        batch_size: Number of images to visualize
        class_names: List of class names (optional)
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    for idx in indices:
        image, target = dataset[idx]
        # Convert image tensor to numpy array for visualization
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()  # HWC
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = np.array(image)
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        S = target.shape[0]
        B = (target.shape[2] - 20) // 5
        C = 20
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image_np)
        for i in range(S):
            for j in range(S):
                cell = target[i, j]
                for b in range(B):
                    conf = cell[C + b * 5 + 4]
                    w = cell[C + b * 5 + 2].item()
                    h = cell[C + b * 5 + 3].item()
                    if conf > 0 and w > 0.01 and h > 0.01:
                        x = (j + cell[C + b * 5 + 0].item()) / S
                        y = (i + cell[C + b * 5 + 1].item()) / S
                        x1 = x - w / 2
                        y1 = y - h / 2
                        x2 = x + w / 2
                        y2 = y + h / 2
                        rect = patches.Rectangle((x1 * image_np.shape[1], y1 * image_np.shape[0]),
                                                 w * image_np.shape[1], h * image_np.shape[0],
                                                 linewidth=3, edgecolor='red', facecolor='none')
                        ax.add_patch(rect)
                        if class_names is not None:
                            class_id = int(torch.argmax(cell[:C]).item())
                            label = f"{class_names[class_id]} ({conf:.2f})"
                            ax.text(x1 * image_np.shape[1], max(0, y1 * image_np.shape[0] - 5),
                                    label, color='yellow', fontsize=12, weight='bold',
                                    bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', pad=0))
        ax.set_title(f"VOC idx={idx}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'voc_gt_{idx}.png'))
        plt.close()