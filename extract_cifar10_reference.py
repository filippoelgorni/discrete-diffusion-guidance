"""Extract CIFAR-10 reference images and save as PNG files.

This script loads CIFAR-10 dataset (train or test split) and saves all images
as PNG files for later metric computation.

Usage:
    # Extract training set
    python extract_cifar10_reference.py \
        --cifar10-path data/cifar10 \
        --output-dir data/cifar10_reference/train \
        --split train

    # Extract test set
    python extract_cifar10_reference.py \
        --cifar10-path data/cifar10 \
        --output-dir data/cifar10_reference/test \
        --split test
"""

import argparse
import json
import os
import pickle
import typing

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_cifar10_from_pickle(root: str, split: str = "train") -> tuple:
    """Load CIFAR-10 directly from pickle files (no downloads).
    
    Args:
        root: Path to CIFAR-10 dataset root (containing cifar-10-batches-py/)
        split: Either "train" or "test"
    
    Returns:
        images: Tensor of shape (N, 3, 32, 32) in range [0, 255]
        labels: Tensor of shape (N,) with class labels
    """
    batches_dir = os.path.join(root, 'cifar-10-batches-py')
    
    if not os.path.exists(batches_dir):
        raise FileNotFoundError(
            f"CIFAR-10 batches directory not found: {batches_dir}\n"
            f"Expected structure: {root}/cifar-10-batches-py/data_batch_* or test_batch"
        )
    
    all_data = []
    all_labels = []
    
    if split == "train":
        # Load all data_batch files
        batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        for batch_file in batch_files:
            batch_path = os.path.join(batches_dir, batch_file)
            if not os.path.exists(batch_path):
                print(f"Warning: {batch_file} not found, skipping...")
                continue
            
            with open(batch_path, 'rb') as f:
                batch_dict = pickle.load(f, encoding='bytes')
            
            # Handle both byte keys and string keys
            if b'data' in batch_dict:
                data = batch_dict[b'data']
                labels = batch_dict[b'labels']
            else:
                data = batch_dict['data']
                labels = batch_dict['labels']
            
            all_data.append(data)
            all_labels.extend(labels)
    else:
        # Load test batch
        test_path = os.path.join(batches_dir, 'test_batch')
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test batch not found: {test_path}")
        
        with open(test_path, 'rb') as f:
            batch_dict = pickle.load(f, encoding='bytes')
        
        if b'data' in batch_dict:
            data = batch_dict[b'data']
            labels = batch_dict[b'labels']
        else:
            data = batch_dict['data']
            labels = batch_dict['labels']
        
        all_data.append(data)
        all_labels.extend(labels)
    
    if not all_data:
        raise ValueError(f"No data loaded from {batches_dir}")
    
    # Concatenate all batches
    data = np.concatenate(all_data, axis=0)
    labels = np.array(all_labels)
    
    print(f"Raw data shape: {data.shape}, dtype: {data.dtype}, range: [{data.min()}, {data.max()}]")
    
    # Handle different data formats:
    # - Standard CIFAR-10: (N, 3072) flat array
    # - Preprocessed subset: (N, 32, 32, 3) already reshaped
    if len(data.shape) == 2:
        # Standard format: (N, 3072) -> reshape to (N, 3, 32, 32) -> transpose to (N, 32, 32, 3)
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    elif len(data.shape) == 4 and data.shape[1:] == (32, 32, 3):
        # Already in correct format (N, 32, 32, 3)
        pass
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    
    # Convert to torch tensors: (N, H, W, C) -> (N, C, H, W)
    images_tensor = torch.from_numpy(data).permute(0, 3, 1, 2).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    print(f"Tensor shape: {images_tensor.shape}, dtype: {images_tensor.dtype}")
    print(f"Image range: min={images_tensor.min():.2f}, max={images_tensor.max():.2f}, mean={images_tensor.mean():.2f}")
    
    return images_tensor, labels_tensor


def save_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    save_dir: str,
    prefix: str = "img",
    class_names: typing.Optional[list] = None,
) -> list:
    """Save images to disk.
    
    Args:
        images: Tensor of images (N, C, H, W) in range [0, 255]
        labels: Tensor of labels (N,)
        save_dir: Directory to save images
        prefix: Prefix for filenames
        class_names: Optional class name list for subfolder naming
    
    Returns:
        List of dicts with image metadata
    """
    os.makedirs(save_dir, exist_ok=True)
    metadata = []
    
    for i, (img, label) in enumerate(tqdm(
            zip(images, labels), 
            desc=f"Saving {prefix} images",
            total=len(images))):
        
        img_uint8 = torch.clamp(img.cpu(), 0, 255).to(torch.uint8)
        img_np = img_uint8.permute(1, 2, 0).numpy()
        
        pil_img = Image.fromarray(img_np)
        class_dir = str(int(label))
        if class_names is not None and 0 <= int(label) < len(class_names):
            class_dir = f"{int(label)}_{class_names[int(label)]}"
        class_dir_path = os.path.join(save_dir, class_dir)
        os.makedirs(class_dir_path, exist_ok=True)
        path = os.path.join(class_dir_path, f"{prefix}_{i:05d}.png")
        pil_img.save(path)
        
        metadata.append({
            "idx": i,
            "path": os.path.relpath(path, save_dir),
            "label": int(label)
        })
    
    return metadata


def main(args):
    # Load CIFAR-10
    print(f"Loading CIFAR-10 {args.split} split from: {args.cifar10_path}")
    images, labels = load_cifar10_from_pickle(args.cifar10_path, args.split)
    
    # Save metadata
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Save images
    print(f"\nSaving images to: {args.output_dir}")
    metadata = save_images(
        images,
        labels,
        args.output_dir,
        prefix=args.split,
        class_names=class_names,
    )
    
    # Count samples per class
    class_counts = {}
    for item in metadata:
        label = item['label']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    metadata_dict = {
        "split": args.split,
        "num_images": len(metadata),
        "source_path": args.cifar10_path,
        "class_names": class_names,
        "class_counts": class_counts,
        "images": metadata
    }
    
    metadata_path = os.path.join(args.output_dir, "reference_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Split: {args.split}")
    print(f"Num images: {len(metadata)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label} ({class_names[label]}): {count}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CIFAR-10 reference images")
    parser.add_argument("--cifar10-path", type=str, required=True,
                        help="Path to CIFAR-10 dataset root")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save extracted images")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Which split to extract (train or test)")
    
    args = parser.parse_args()
    main(args)
