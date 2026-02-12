#!/usr/bin/env python3
"""
Create a balanced subset of CIFAR-10 with deterministic selection.

Usage:
    python preprocess_cifar10_subset.py \
        --source-dir /path/to/original/cifar10 \
        --output-dir /path/to/output/cifar10_subset \
        --fraction 0.1 \
        --categories 0 1 2 3 4
"""
import argparse
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets


def create_balanced_subset(source_dir, output_dir, fraction=0.1, categories=None, seed=42):
    """
    Create a deterministic, balanced subset of CIFAR-10.
    
    Args:
        source_dir: Path to original CIFAR-10 directory
        output_dir: Path where to save the subset
        fraction: Fraction of data to keep (0.0-1.0)
        categories: List of category indices to include. If None, uses all.
        seed: Random seed for deterministic selection
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original CIFAR-10
    dataset = torchvision.datasets.CIFAR10(
        root=source_dir, train=True, download=True)
    
    data = dataset.data  # Shape: (50000, 32, 32, 3)
    targets = np.array(dataset.targets)
    
    if categories is not None:
        # Filter by specified categories
        mask = np.isin(targets, categories)
        indices = np.where(mask)[0]
    else:
        indices = np.arange(len(targets))
    
    # For balanced subset: sample uniformly from each category
    unique_categories = np.unique(targets[indices])
    selected_indices = []
    
    for cat in unique_categories:
        cat_mask = targets[indices] == cat
        cat_indices = indices[cat_mask]
        num_samples = max(1, int(len(cat_indices) * fraction))
        
        # Deterministic selection using seed
        selected = np.random.choice(
            cat_indices, size=num_samples, replace=False)
        selected_indices.extend(selected)
    
    selected_indices = np.array(selected_indices)
    selected_indices.sort()  # Keep order consistent
    
    # Create subset data
    subset_data = data[selected_indices]
    subset_targets = targets[selected_indices]
    
    # Save in CIFAR-10 format (batches)
    batches_dir = output_dir / 'cifar-10-batches-py'
    batches_dir.mkdir(parents=True, exist_ok=True)

    # Save five training batch files (data_batch_1..5)
    batch_splits = np.array_split(np.arange(len(subset_data)), 5)
    for batch_idx, batch_indices in enumerate(batch_splits, start=1):
        batch_data = subset_data[batch_indices]
        batch_targets = subset_targets[batch_indices]
        batch_dict = {
            'batch_label': f'training batch {batch_idx}',
            'labels': batch_targets.tolist(),
            'data': batch_data,
            'filenames': [f'img_{i}.png' for i in batch_indices.tolist()]
        }
        with open(batches_dir / f'data_batch_{batch_idx}', 'wb') as f:
            pickle.dump(batch_dict, f)
    
    # Save test set (unchanged)
    test_dataset = torchvision.datasets.CIFAR10(
        root=source_dir, train=False, download=True)
    test_batch_dict = {
        'batch_label': 'test batch',
        'labels': test_dataset.targets,
        'data': test_dataset.data,
        'filenames': [f'img_{i}.png' for i in range(len(test_dataset.data))]
    }
    
    with open(batches_dir / 'test_batch', 'wb') as f:
        pickle.dump(test_batch_dict, f)
    
    # Save meta
    meta_dict = {
        'label_names': dataset.classes,
        'num_cases_per_batch': int(max(len(x) for x in batch_splits)),
        'num_vis': 3072
    }
    
    with open(batches_dir / 'batches.meta', 'wb') as f:
        pickle.dump(meta_dict, f)
    
    print(f"Created subset: {len(subset_data)} samples ({fraction*100:.1f}%)")
    print(f"Categories: {sorted(unique_categories.tolist())}")
    print(f"Output: {output_dir}")
    
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create balanced CIFAR-10 subset')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Path to original CIFAR-10')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Path to save subset')
    parser.add_argument('--fraction', type=float, default=0.1,
                       help='Fraction of data to keep (0.0-1.0)')
    parser.add_argument('--categories', type=int, nargs='*',
                       help='Categories to include (e.g., 0 1 2 3 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for deterministic selection')
    
    args = parser.parse_args()
    
    create_balanced_subset(
        args.source_dir,
        args.output_dir,
        fraction=args.fraction,
        categories=args.categories if args.categories else None,
        seed=args.seed
    )
