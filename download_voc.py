#!/usr/bin/env python3
"""
Main script to download, prepare, and preprocess the PASCAL VOC 2012 dataset for YOLOv1.
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import shutil
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from src
from preprocessor import VOCPreprocessor

# VOC 2012 dataset URLs
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

def download_file(url, filename):
    """Download a file with progress bar."""
    def report_progress(block_num, block_size, total_size):
        pbar.update(block_size)

    print(f"Downloading {filename}...")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as pbar:
        urllib.request.urlretrieve(url, filename, reporthook=report_progress)

def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        'data/VOC2012',
        'checkpoints',
        'logs',
        'results'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def extract_dataset(filename):
    """Extract the downloaded dataset."""
    print("Extracting dataset...")
    with tarfile.open(filename, 'r') as tar:
        tar.extractall(path='data')

    # Move contents to the correct location
    voc_path = 'data/VOCdevkit/VOC2012'
    if os.path.exists(voc_path):
        for item in os.listdir(voc_path):
            src = os.path.join(voc_path, item)
            dst = os.path.join('data/VOC2012', item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree('data/VOCdevkit')
        print("Dataset extracted and organized successfully!")
    else:
        print("Error: Could not find extracted VOC2012 directory")
        sys.exit(1)

def download_and_prepare_dataset():
    """Download and prepare the VOC dataset."""
    # Create directory structure
    create_directory_structure()

    # Download and extract dataset
    dataset_file = 'data/VOCtrainval_11-May-2012.tar'
    if not os.path.exists(dataset_file):
        download_file(VOC_URL, dataset_file)

    if not os.path.exists('data/VOC2012/JPEGImages'):
        extract_dataset(dataset_file)
    else:
        print("Dataset already exists. Skipping download and extraction.")

    print("\nDataset preparation completed!")
    print("Dataset location: data/VOC2012/")
    print("\nDirectory structure:")
    print("- data/VOC2012/")
    print("  - JPEGImages/     (training images)")
    print("  - Annotations/    (XML annotation files)")
    print("  - ImageSets/      (train/val splits)")
    print("- checkpoints/      (for saving model checkpoints)")
    print("- logs/            (for training logs)")
    print("- results/         (for evaluation results)")

def preprocess_dataset(args):
    """Preprocess the VOC dataset for YOLOv1."""
    # Create preprocessor
    preprocessor = VOCPreprocessor(
        voc_root=args.voc_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        grid_size=args.grid_size
    )

    # Process dataset
    if args.split == 'all' or args.split == 'train':
        print("\n=== Processing Training Set ===")
        preprocessor.process_dataset(split='train')

    if args.split == 'all' or args.split == 'val':
        print("\n=== Processing Validation Set ===")
        preprocessor.process_dataset(split='val')

    print("\n=== Dataset Preprocessing Complete ===")
    print(f"Processed data saved to: {args.output_dir}")
    print(f"  - Images: {args.output_dir}/images/")
    print(f"  - Labels: {args.output_dir}/labels/")

def main():
    """Main function to download, prepare, and preprocess the dataset."""
    parser = argparse.ArgumentParser(description='Download and preprocess PASCAL VOC dataset for YOLOv1')
    parser.add_argument('--voc_root', type=str, default='data/VOC2012',
                        help='Root directory of VOC dataset')
    parser.add_argument('--output_dir', type=str, default='data/yolo_voc2012',
                        help='Directory to save processed data')
    parser.add_argument('--image_size', type=int, default=448,
                        help='Size to resize images to')
    parser.add_argument('--grid_size', type=int, default=7,
                        help='Size of the grid')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'all'],
                        help='Dataset split to process')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading and extracting the dataset')

    args = parser.parse_args()

    # Download and prepare dataset if needed
    if not args.skip_download:
        download_and_prepare_dataset()
    
    # Preprocess the dataset
    preprocess_dataset(args)

if __name__ == "__main__":
    main() 