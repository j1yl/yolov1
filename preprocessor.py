import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm


class VOCPreprocessor:
    """
    Preprocessor for converting PASCAL VOC dataset to YOLOv1 format.
    """

    # VOC classes
    VOC_CLASSES = [
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

    def __init__(self, voc_root, output_dir, image_size=448, grid_size=7):
        """
        Initialize the preprocessor.

        Args:
            voc_root (str): Root directory of VOC dataset
            output_dir (str): Directory to save processed data
            image_size (int): Size to resize images to
            grid_size (int): Size of the grid for YOLOv1
        """
        self.voc_root = voc_root
        self.output_dir = output_dir
        self.image_size = image_size
        self.grid_size = grid_size

        # Create output directories
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        # Create class mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VOC_CLASSES)}

    def _parse_voc_xml(self, xml_path):
        """
        Parse VOC XML annotation file.

        Args:
            xml_path (str): Path to XML annotation file

        Returns:
            list: List of dictionaries containing object information
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        objects = []
        for obj in root.findall("object"):
            # Get class name
            class_name = obj.find("name").text

            # Get bounding box
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Convert to YOLO format (normalized coordinates)
            x_center = (xmin + xmax) / (2.0 * width)
            y_center = (ymin + ymax) / (2.0 * height)
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            objects.append(
                {
                    "class": class_name,
                    "class_idx": self.class_to_idx[class_name],
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": box_width,
                    "height": box_height,
                }
            )

        return objects

    def _create_yolo_label(self, objects):
        """
        Create YOLO format label string.

        Args:
            objects (list): List of object dictionaries

        Returns:
            str: YOLO format label string
        """
        label_lines = []
        for obj in objects:
            # Format: <class> <x_center> <y_center> <width> <height>
            line = f"{obj['class_idx']} {obj['x_center']:.6f} {obj['y_center']:.6f} {obj['width']:.6f} {obj['height']:.6f}"
            label_lines.append(line)

        return "\n".join(label_lines)

    def _process_image(self, image_path, output_image_path):
        """
        Process and resize image.

        Args:
            image_path (str): Path to input image
            output_image_path (str): Path to save processed image
        """
        # Open and resize image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize image
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

            # Save processed image
            img.save(output_image_path)

    def process_dataset(self, split="train"):
        """
        Process the dataset for a specific split.

        Args:
            split (str): Dataset split to process ('train' or 'val')
        """
        # Get image list
        imagesets_dir = os.path.join(self.voc_root, "ImageSets", "Main")
        with open(os.path.join(imagesets_dir, f"{split}.txt"), "r") as f:
            image_ids = [line.strip() for line in f.readlines()]

        print(f"Processing {split} set with {len(image_ids)} images...")

        # Process each image
        for img_id in tqdm(image_ids, desc=f"Processing {split} set"):
            # Paths
            img_path = os.path.join(self.voc_root, "JPEGImages", f"{img_id}.jpg")
            xml_path = os.path.join(self.voc_root, "Annotations", f"{img_id}.xml")
            output_img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
            output_label_path = os.path.join(self.labels_dir, f"{img_id}.txt")

            # Process image
            self._process_image(img_path, output_img_path)

            # Process annotation
            objects = self._parse_voc_xml(xml_path)
            label_str = self._create_yolo_label(objects)

            # Save label
            with open(output_label_path, "w") as f:
                f.write(label_str)

    def create_dataset_info(self):
        """
        Create dataset info file with class names.
        """
        info_path = os.path.join(self.output_dir, "classes.txt")
        with open(info_path, "w") as f:
            for cls in self.VOC_CLASSES:
                f.write(f"{cls}\n")

        print(f"Dataset info saved to: {info_path}")
