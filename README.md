# YOLO v1 Implementation

This is a PyTorch implementation of YOLO v1 for object detection on the Pascal VOC dataset.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and preprocess the Pascal VOC dataset:
```bash
python download_voc.py
```

## Training

To train the model:

```bash
python train.py
```

The training script will:
1. Load and preprocess the Pascal VOC dataset
2. Initialize the YOLO v1 model
3. Train for the specified number of epochs
4. Save checkpoints after each epoch

## Inference

To run inference on an image:

```bash
python predict.py
```

Make sure to:
1. Update the `MODEL_PATH` in `predict.py` to point to your trained model
2. Update the `IMAGE_PATH` to point to your test image

## Model Architecture

The implementation follows the original YOLO v1 architecture:
- Input image size: 448x448
- Grid size: 7x7
- Number of bounding boxes per cell: 2
- Number of classes: 20 (Pascal VOC classes)

## Dataset

The Pascal VOC 2012 dataset is used for training. The dataset includes:
- 20 object classes
- Training and validation splits
- Bounding box annotations
- Image segmentation masks (not used in this implementation)

## License

This implementation is for educational purposes only. The Pascal VOC dataset is subject to its own license terms. 