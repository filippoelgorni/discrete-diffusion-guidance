"""Script to create a subset of CIFAR-10 and save it in a format compatible
with DiscreteCIFAR10 (numpy .npz files).

Usage:
  python preprocess_cifar10_subset.py \
    --source_dir <path/to/full/cifar10> \
    --output_dir <path/to/output/subset> \
    --fraction 0.002 \
    --seed 42
"""
import argparse
import os

import numpy as np
import torchvision


def create_cifar10_subset(source_dir, output_dir, fraction, seed=42):
  """Download CIFAR-10 (if needed) and save a random subset as .npz files.

  The output directory will contain:
    train.npz  - dict with keys 'data' (N, 32, 32, 3) uint8 and 'targets' (N,)
    test.npz   - same format for the test split

  Args:
    source_dir: Directory to download / look for the full CIFAR-10 dataset.
    output_dir: Directory where the subset .npz files will be written.
    fraction:   Fraction of each split to keep (e.g. 0.002 for 0.2%).
    seed:       Random seed for reproducibility.
  """
  rng = np.random.default_rng(seed)

  os.makedirs(source_dir, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)

  for is_train in (True, False):
    split = 'train' if is_train else 'test'
    full_ds = torchvision.datasets.CIFAR10(
      root=source_dir, train=is_train, download=True)

    n_total = len(full_ds)
    n_subset = max(1, int(n_total * fraction))
    indices = rng.choice(n_total, size=n_subset, replace=False)

    data = full_ds.data[indices]                        # (N, 32, 32, 3) uint8
    targets = np.array(full_ds.targets)[indices]        # (N,)

    out_path = os.path.join(output_dir, f'{split}.npz')
    np.savez(out_path, data=data, targets=targets)
    print(f'[{split}] {n_subset}/{n_total} samples saved to {out_path}')


def main():
  parser = argparse.ArgumentParser(
    description='Create a CIFAR-10 subset compatible with DiscreteCIFAR10.')
  parser.add_argument(
    '--source_dir', required=True,
    help='Directory to download / find the full CIFAR-10 dataset.')
  parser.add_argument(
    '--output_dir', required=True,
    help='Directory where the subset .npz files will be saved.')
  parser.add_argument(
    '--fraction', type=float, default=0.002,
    help='Fraction of each split to keep (default: 0.002).')
  parser.add_argument(
    '--seed', type=int, default=42,
    help='Random seed for reproducibility (default: 42).')
  args = parser.parse_args()

  create_cifar10_subset(
    source_dir=args.source_dir,
    output_dir=args.output_dir,
    fraction=args.fraction,
    seed=args.seed,
  )


if __name__ == '__main__':
  main()
