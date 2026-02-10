"""Compute memorization metrics from saved generated and reference images.

This script loads previously generated images and CIFAR-10 reference images,
then computes memorization metrics. Can be run locally without GPU.

Usage:
    python compute_cifar10_metrics.py \
        --generated-dir outputs/cifar10/mdlm_v1/generated_samples_final \
        --reference-dir data/cifar10_reference/train \
        --output-dir outputs/cifar10/mdlm_v1/eval_final \
        --mem-threshold 0.333333
"""

import argparse
import json
import os
import typing

import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm


def build_image_paths(
    image_dir: str,
    metadata: typing.Optional[dict],
    metadata_list_key: str,
) -> typing.List[str]:
    """Build ordered list of image paths from metadata or directory scan."""
    if metadata and metadata_list_key in metadata:
        items = metadata[metadata_list_key]
        if items and isinstance(items, list):
            if all(isinstance(item, dict) and "idx" in item for item in items):
                items = sorted(items, key=lambda x: x["idx"])
            paths = []
            for item in items:
                rel_path = item.get("path")
                if rel_path is None:
                    continue
                if os.path.isabs(rel_path):
                    paths.append(rel_path)
                else:
                    paths.append(os.path.join(image_dir, rel_path))
            if paths:
                return paths

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for name in files:
            if name.endswith(".png"):
                image_paths.append(os.path.join(root, name))
    image_paths.sort()
    return image_paths


def load_images_from_paths(
    image_paths: typing.List[str],
    label: str,
) -> torch.Tensor:
    """Load PNG images from provided paths.
    
    Args:
        image_paths: Ordered list of PNG image paths
        label: Label for progress output
    
    Returns:
        images: Tensor of shape (N, 3, 32, 32) in range [0, 255]
    """
    images = []
    for img_path in tqdm(image_paths, desc=f"Loading images from {label}"):
        pil_img = Image.open(img_path)
        img_array = np.array(pil_img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        images.append(img_tensor)

    images_tensor = torch.stack(images)
    
    print(f"Loaded {len(images_tensor)} images from {label}")
    print(f"Shape: {images_tensor.shape}")
    print(f"Range: min={images_tensor.min():.2f}, max={images_tensor.max():.2f}, mean={images_tensor.mean():.2f}")
    
    return images_tensor


def compute_memorization(
    generated: torch.Tensor,
    reference: torch.Tensor,
    k: float = 1/3,
) -> typing.Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute memorization metric f_mem.
    
    A sample is memorized if:
        ||x - a^μ1||_2 / ||x - a^μ2||_2 < k
    
    Args:
        generated: Generated samples (N, C, H, W)
        reference: Reference images (M, C, H, W)
        k: Threshold for memorization (default 1/3)
    
    Returns:
        f_mem: Fraction of memorized samples
        nearest_indices: Index of nearest neighbor for each sample
        memorization_ratios: Ratio for each sample
    """
    # Validate input shapes
    assert len(generated.shape) == 4 and generated.shape[1:] == (3, 32, 32), \
        f"Generated samples shape {generated.shape} != (N, 3, 32, 32)"
    assert len(reference.shape) == 4 and reference.shape[1:] == (3, 32, 32), \
        f"Reference images shape {reference.shape} != (M, 3, 32, 32)"
    
    # Flatten images
    gen_flat = generated.view(generated.shape[0], -1).numpy()
    ref_flat = reference.view(reference.shape[0], -1).numpy()
    
    print("Computing pairwise distances...")
    # Process in chunks to avoid OOM
    chunk_size = 500
    nearest_indices = []
    second_nearest_indices = []
    memorization_ratios = []
    
    for i in tqdm(range(0, len(gen_flat), chunk_size), desc="Finding nearest neighbors"):
        chunk = gen_flat[i:i + chunk_size]
        distances = cdist(chunk, ref_flat, metric='euclidean')
        
        # Get indices of 2 nearest neighbors
        sorted_indices = np.argsort(distances, axis=1)
        nearest_idx = sorted_indices[:, 0]
        second_nearest_idx = sorted_indices[:, 1]
        
        # Get distances to nearest and second nearest
        d1 = distances[np.arange(len(chunk)), nearest_idx]
        d2 = distances[np.arange(len(chunk)), second_nearest_idx]
        
        # Compute ratio (avoid division by zero)
        ratio = d1 / (d2 + 1e-8)
        
        nearest_indices.extend(nearest_idx.tolist())
        second_nearest_indices.extend(second_nearest_idx.tolist())
        memorization_ratios.extend(ratio.tolist())
    
    nearest_indices = np.array(nearest_indices)
    memorization_ratios = np.array(memorization_ratios)
    
    # Count memorized samples
    memorized = memorization_ratios < k
    f_mem = memorized.sum() / len(memorized)
    
    return f_mem, nearest_indices, memorization_ratios


def labels_from_metadata(
    metadata: typing.Optional[dict],
    metadata_list_key: str,
    label_key: str,
) -> typing.Optional[typing.List[typing.Optional[int]]]:
    """Extract labels list aligned with metadata ordering."""
    if not metadata or metadata_list_key not in metadata:
        return None

    items = metadata[metadata_list_key]
    if not items or not isinstance(items, list):
        return None

    if all(isinstance(item, dict) and "idx" in item for item in items):
        items = sorted(items, key=lambda x: x["idx"])

    labels = []
    for item in items:
        labels.append(item.get(label_key))
    return labels


def compute_memorization_by_class(
    generated_paths: typing.List[str],
    reference_paths: typing.List[str],
    gen_metadata: typing.Optional[dict],
    ref_metadata: typing.Optional[dict],
    k: float,
) -> typing.Tuple[float, np.ndarray, np.ndarray, dict]:
    """Compute memorization per class, comparing within class folders."""
    gen_labels = labels_from_metadata(
        gen_metadata, metadata_list_key="sample_info", label_key="condition")
    ref_labels = labels_from_metadata(
        ref_metadata, metadata_list_key="images", label_key="label")

    if gen_labels is None or ref_labels is None:
        return None, None, None, {}

    num_generated = len(generated_paths)
    nearest_indices = np.zeros(num_generated, dtype=int)
    mem_ratios = np.zeros(num_generated, dtype=float)
    per_class_metrics = {}

    # Build index groups for generated and reference sets
    gen_indices_by_label = {}
    for i, label in enumerate(gen_labels):
        gen_indices_by_label.setdefault(label, []).append(i)

    ref_indices_by_label = {}
    for i, label in enumerate(ref_labels):
        ref_indices_by_label.setdefault(label, []).append(i)

    for label, gen_indices in gen_indices_by_label.items():
        gen_paths = [generated_paths[i] for i in gen_indices]
        ref_indices = ref_indices_by_label.get(label, [])

        if ref_indices:
            ref_paths = [reference_paths[i] for i in ref_indices]
            ref_global_indices = ref_indices
        else:
            ref_paths = reference_paths
            ref_global_indices = list(range(len(reference_paths)))

        gen_images = load_images_from_paths(gen_paths, f"generated class {label}")
        ref_images = load_images_from_paths(ref_paths, f"reference class {label}")

        f_mem, nn_local_indices, ratios = compute_memorization(gen_images, ref_images, k=k)
        nn_global_indices = [ref_global_indices[i] for i in nn_local_indices]

        for local_idx, gen_idx in enumerate(gen_indices):
            nearest_indices[gen_idx] = nn_global_indices[local_idx]
            mem_ratios[gen_idx] = ratios[local_idx]

        per_class_metrics[str(label)] = {
            "num_generated": int(len(gen_paths)),
            "num_reference": int(len(ref_paths)),
            "f_mem": float(f_mem),
            "f_mem_percent": float(f_mem * 100),
        }

    f_mem_total = float((mem_ratios < k).sum() / len(mem_ratios))
    return f_mem_total, nearest_indices, mem_ratios, per_class_metrics


def save_nearest_neighbor_images(
    generated_paths: typing.List[str],
    reference_paths: typing.List[str],
    nearest_indices: np.ndarray,
    memorization_ratios: np.ndarray,
    output_dir: str,
    mem_threshold: float,
    reference_metadata: typing.Optional[dict] = None,
):
    """Copy nearest neighbor reference images next to generated images.
    
    Organizes images into memorized/non-memorized subdirectories.
    """
    assert len(generated_paths) == len(nearest_indices), \
        f"Number of generated images ({len(generated_paths)}) != number of NN indices ({len(nearest_indices)})"
    
    # Create output directories
    gen_memorized_dir = os.path.join(output_dir, "generated_images", "memorized")
    gen_non_memorized_dir = os.path.join(output_dir, "generated_images", "non_memorized")
    nn_memorized_dir = os.path.join(output_dir, "nearest_neighbors", "memorized")
    nn_non_memorized_dir = os.path.join(output_dir, "nearest_neighbors", "non_memorized")
    
    for d in [gen_memorized_dir, gen_non_memorized_dir, nn_memorized_dir, nn_non_memorized_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Copy images and create metadata
    nn_info = []
    
        for i, (gen_path, nn_idx, ratio) in enumerate(tqdm(
            zip(generated_paths, nearest_indices, memorization_ratios),
            desc="Organizing images",
            total=len(generated_paths))):
        
        is_memorized = ratio < mem_threshold
        
        # Copy generated image
        gen_src = gen_path
        gen_file = os.path.basename(gen_path)
        if is_memorized:
            gen_dst = os.path.join(gen_memorized_dir, gen_file)
        else:
            gen_dst = os.path.join(gen_non_memorized_dir, gen_file)
        
        # Use symlink if possible, otherwise copy
        try:
            if os.path.exists(gen_dst):
                os.remove(gen_dst)
            os.symlink(os.path.abspath(gen_src), gen_dst)
        except:
            import shutil
            shutil.copy2(gen_src, gen_dst)
        
        # Copy nearest neighbor reference image
        ref_src = reference_paths[nn_idx]
        
        if is_memorized:
            nn_dst = os.path.join(nn_memorized_dir, f"nn_{i:05d}_refidx{nn_idx}.png")
        else:
            nn_dst = os.path.join(nn_non_memorized_dir, f"nn_{i:05d}_refidx{nn_idx}.png")
        
        try:
            if os.path.exists(nn_dst):
                os.remove(nn_dst)
            os.symlink(os.path.abspath(ref_src), nn_dst)
        except:
            import shutil
            shutil.copy2(ref_src, nn_dst)
        
        # Get label if available
        ref_label = None
        if reference_metadata and 'images' in reference_metadata:
            if nn_idx < len(reference_metadata['images']):
                ref_label = reference_metadata['images'][nn_idx].get('label')
        
        nn_info.append({
            "generated_idx": i,
            "generated_path": gen_dst,
            "nearest_neighbor_idx": int(nn_idx),
            "nearest_neighbor_path": nn_dst,
            "nearest_neighbor_label": ref_label,
            "memorization_ratio": float(ratio),
            "is_memorized": bool(is_memorized),
        })
    
    return nn_info


def main(args):
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CIFAR-10 Memorization Metric Computation")
    print("=" * 60)
    print(f"Generated images: {args.generated_dir}")
    print(f"Reference images: {args.reference_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Memorization threshold (k): {args.mem_threshold}")
    print("=" * 60)
    
    # Load metadata if available
    gen_metadata_path = os.path.join(args.generated_dir, "generation_metadata.json")
    gen_metadata = None
    if os.path.exists(gen_metadata_path):
        with open(gen_metadata_path, 'r') as f:
            gen_metadata = json.load(f)
        print(f"\nLoaded generation metadata from: {gen_metadata_path}")
    
    ref_metadata_path = os.path.join(args.reference_dir, "reference_metadata.json")
    ref_metadata = None
    if os.path.exists(ref_metadata_path):
        with open(ref_metadata_path, 'r') as f:
            ref_metadata = json.load(f)
        print(f"Loaded reference metadata from: {ref_metadata_path}")
    
    # Load images
    print("\n=== Loading images ===")
    generated_paths = build_image_paths(
        args.generated_dir,
        gen_metadata,
        metadata_list_key="sample_info",
    )
    reference_paths = build_image_paths(
        args.reference_dir,
        ref_metadata,
        metadata_list_key="images",
    )
    
    # Compute memorization
    print(f"\n=== Computing memorization (k={args.mem_threshold}) ===")
    f_mem, nearest_indices, mem_ratios, per_class_metrics = compute_memorization_by_class(
        generated_paths,
        reference_paths,
        gen_metadata,
        ref_metadata,
        k=args.mem_threshold,
    )

    if f_mem is None:
        generated_images = load_images_from_paths(generated_paths, "generated")
        reference_images = load_images_from_paths(reference_paths, "reference")
        f_mem, nearest_indices, mem_ratios = compute_memorization(
            generated_images, reference_images, k=args.mem_threshold)
        per_class_metrics = {}
    else:
        generated_images = load_images_from_paths(generated_paths, "generated")
        reference_images = load_images_from_paths(reference_paths, "reference")
    print(f"Memorization f_mem: {f_mem:.4f} ({f_mem*100:.2f}%)")
    
    # Determine which samples are memorized
    is_memorized = mem_ratios < args.mem_threshold
    num_memorized = is_memorized.sum()
    num_non_memorized = (~is_memorized).sum()
    
    print(f"Memorized: {num_memorized}, Non-memorized: {num_non_memorized}")
    
    # Save organized images and nearest neighbors
    if args.save_images:
        print("\n=== Organizing images ===")
        nn_info = save_nearest_neighbor_images(
            generated_paths,
            reference_paths,
            nearest_indices,
            mem_ratios,
            args.output_dir,
            args.mem_threshold,
            ref_metadata,
        )
    else:
        nn_info = [
            {
                "generated_idx": i,
                "nearest_neighbor_idx": int(nearest_indices[i]),
                "memorization_ratio": float(mem_ratios[i]),
                "is_memorized": bool(is_memorized[i]),
            }
            for i in range(len(nearest_indices))
        ]
    
    # Compile results
    results = {
        "generated_dir": args.generated_dir,
        "reference_dir": args.reference_dir,
        "num_generated": len(generated_images),
        "num_reference": len(reference_images),
        "metrics": {
            "f_mem": float(f_mem),
            "f_mem_percent": float(f_mem * 100),
            "memorization_threshold_k": args.mem_threshold,
            "num_memorized": int(num_memorized),
            "num_non_memorized": int(num_non_memorized),
        },
        "per_class_metrics": per_class_metrics,
        "generation_metadata": gen_metadata,
        "reference_metadata": ref_metadata,
        "nearest_neighbor_info": nn_info,
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, "memorization_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Results saved to {results_path} ===")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Num generated samples: {len(generated_images)}")
    print(f"Num reference images: {len(reference_images)}")
    print(f"Memorization (f_mem): {f_mem*100:.2f}%")
    print(f"Memorized samples: {num_memorized}/{len(generated_images)}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute memorization metrics from saved images")
    parser.add_argument("--generated-dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--reference-dir", type=str, required=True,
                        help="Directory containing reference images")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results and organized images")
    parser.add_argument("--mem-threshold", type=float, default=1/3,
                        help="Threshold k for memorization detection")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false",
                        help="Don't organize and save images, only compute metrics")
    parser.set_defaults(save_images=True)
    
    args = parser.parse_args()
    main(args)
