"""CIFAR-10 Evaluation Script.

Computes:
- Memorization metric f_mem (percentage of memorized samples)
- Nearest neighbor analysis

A sample is considered memorized if:
  ||x - a^μ1||_2 / ||x - a^μ2||_2 < k
where a^μ1 is the nearest neighbor, a^μ2 is the 2nd nearest neighbor,
and k is a threshold (default 1/3).
"""

import argparse
import json
import os
import typing

import einops
import lightning as L
import numpy as np
import omegaconf
import torch
import torchvision
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm

import dataloader
import diffusion


def load_cifar10_train(root: str) -> torch.Tensor:
    """Load CIFAR-10 training set as tensor."""
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True,
        transform=torchvision.transforms.ToTensor())
    images = []
    labels = []
    for img, label in tqdm(dataset, desc="Loading CIFAR-10 train"):
        images.append(img)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def generate_samples(
    model: diffusion.Diffusion,
    num_samples: int,
    batch_size: int = 64,
) -> torch.Tensor:
    """Generate samples from the diffusion model using same procedure as training."""
    model.eval()
    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - len(samples) * batch_size if samples else batch_size)
        model.config.sampling.batch_size = current_batch_size
        
        # Use the same sampling procedure as training validation
        with torch.no_grad():
            # model.sample() returns raw tokens, need to decode them
            sample_tokens = model.sample()
            # batch_decode converts tokens to images
            decoded = model.tokenizer.batch_decode(sample_tokens)
            # decoded is already in [0, 1] range and on CPU
            samples.append(decoded)
    
    all_samples = torch.cat(samples, dim=0)[:num_samples]
    return all_samples


def compute_memorization(
    generated: torch.Tensor,
    train_images: torch.Tensor,
    k: float = 1/3,
) -> typing.Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute memorization metric f_mem.
    
    A sample is memorized if:
        ||x - a^μ1||_2 / ||x - a^μ2||_2 < k
    
    Args:
        generated: Generated samples (N, C, H, W)
        train_images: Training images (M, C, H, W)
        k: Threshold for memorization (default 1/3)
    
    Returns:
        f_mem: Fraction of memorized samples
        nearest_indices: Index of nearest neighbor for each sample
        memorization_ratios: Ratio for each sample
    """
    # Flatten images
    gen_flat = generated.cpu().view(generated.shape[0], -1).numpy()
    train_flat = train_images.cpu().view(train_images.shape[0], -1).numpy()
    
    print("Computing pairwise distances...")
    # Compute L2 distances (this can be memory intensive)
    # Process in chunks to avoid OOM
    chunk_size = 500
    nearest_indices = []
    second_nearest_indices = []
    memorization_ratios = []
    
    for i in tqdm(range(0, len(gen_flat), chunk_size), desc="Finding nearest neighbors"):
        chunk = gen_flat[i:i + chunk_size]
        distances = cdist(chunk, train_flat, metric='euclidean')
        
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




def save_images(
    images: torch.Tensor,
    save_dir: str,
    prefix: str = "sample",
    indices: typing.Optional[typing.List[int]] = None,
) -> typing.List[str]:
    """Save images to disk.
    
    Args:
        images: Tensor of images (N, C, H, W) in range [0, 1] or [0, 255]
        save_dir: Directory to save images
        prefix: Prefix for filenames
        indices: Optional list of original indices (for naming)
    
    Returns:
        List of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    
    for i, img in enumerate(tqdm(images, desc=f"Saving {prefix} images")):
        # Use original index if provided, otherwise use loop index
        idx = indices[i] if indices is not None else i
        
        # Convert to PIL
        # Handle both [0, 1] and [0, 255] ranges
        if img.max() <= 1.0:
            img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        
        pil_img = Image.fromarray(img_np)
        
        path = os.path.join(save_dir, f"{prefix}_{idx:05d}.png")
        pil_img.save(path)
        paths.append(path)
    
    return paths




def main(args):
    # Reproducibility
    L.seed_everything(args.seed)
    
    # Paths
    checkpoint_dir = os.path.join(args.outputs_dir, args.run_name)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoints", args.checkpoint)
    config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
    
    # Output directory for evaluation
    eval_dir = os.path.join(checkpoint_dir, f"eval_step{args.eval_step}")
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Saving results to: {eval_dir}")
    
    # Load config
    config = omegaconf.OmegaConf.load(config_path)
    
    # Override sampling settings
    config.sampling.steps = args.sampling_steps
    
    # Load tokenizer and model
    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False,
    )
    model = model.to('cuda')
    model.eval()
    
    # Load CIFAR-10 training data
    print("\n=== Loading CIFAR-10 training data ===")
    train_images, train_labels = load_cifar10_train(args.cifar10_path)
    print(f"Loaded {len(train_images)} training images")
    
    # Generate samples
    print(f"\n=== Generating {args.num_samples} samples ===")
    generated_samples = generate_samples(
        model, args.num_samples, batch_size=args.batch_size)
    print(f"Generated {len(generated_samples)} samples")
    
    # Compute memorization FIRST (before saving images)
    print(f"\n=== Computing memorization (k={args.mem_threshold}) ===")
    f_mem, nearest_indices, mem_ratios = compute_memorization(
        generated_samples, train_images, k=args.mem_threshold)
    print(f"Memorization f_mem: {f_mem:.4f} ({f_mem*100:.2f}%)")
    
    # Determine which samples are memorized
    is_memorized = mem_ratios < args.mem_threshold
    memorized_indices = np.where(is_memorized)[0].tolist()
    non_memorized_indices = np.where(~is_memorized)[0].tolist()
    
    print(f"Memorized: {len(memorized_indices)}, Non-memorized: {len(non_memorized_indices)}")
    
    # Save generated images to separate directories based on memorization
    print("\n=== Saving generated images ===")
    gen_images_dir = os.path.join(eval_dir, "generated_images")
    memorized_dir = os.path.join(gen_images_dir, "memorized")
    non_memorized_dir = os.path.join(gen_images_dir, "non_memorized")
    
    # Track paths for all samples (in order)
    all_gen_paths = [None] * len(generated_samples)
    
    # Save memorized images
    if len(memorized_indices) > 0:
        memorized_samples = generated_samples[memorized_indices]
        mem_paths = save_images(memorized_samples, memorized_dir, prefix="gen", indices=memorized_indices)
        for i, idx in enumerate(memorized_indices):
            all_gen_paths[idx] = mem_paths[i]
    
    # Save non-memorized images
    if len(non_memorized_indices) > 0:
        non_memorized_samples = generated_samples[non_memorized_indices]
        non_mem_paths = save_images(non_memorized_samples, non_memorized_dir, prefix="gen", indices=non_memorized_indices)
        for i, idx in enumerate(non_memorized_indices):
            all_gen_paths[idx] = non_mem_paths[i]
    
    # Save nearest neighbor info and images
    print("\n=== Saving nearest neighbor images ===")
    nn_images_dir = os.path.join(eval_dir, "nearest_neighbors")
    nn_memorized_dir = os.path.join(nn_images_dir, "memorized")
    nn_non_memorized_dir = os.path.join(nn_images_dir, "non_memorized")
    os.makedirs(nn_memorized_dir, exist_ok=True)
    os.makedirs(nn_non_memorized_dir, exist_ok=True)
    
    nn_info = []
    for i, (gen_img, nn_idx, ratio) in enumerate(tqdm(
            zip(generated_samples, nearest_indices, mem_ratios),
            desc="Saving NN info", total=len(generated_samples))):
        
        nn_img = train_images[nn_idx]
        nn_label = train_labels[nn_idx].item()
        sample_is_memorized = ratio < args.mem_threshold
        
        # Save nearest neighbor image to appropriate directory
        # Handle both [0, 1] and [0, 255] ranges
        if nn_img.max() <= 1.0:
            nn_img_np = (nn_img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            nn_img_np = nn_img.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        
        nn_pil = Image.fromarray(nn_img_np)
        
        if sample_is_memorized:
            nn_subdir = nn_memorized_dir
        else:
            nn_subdir = nn_non_memorized_dir
        
        nn_path = os.path.join(nn_subdir, f"nn_{i:05d}_trainidx{nn_idx}.png")
        nn_pil.save(nn_path)
        
        nn_info.append({
            "generated_idx": i,
            "generated_path": all_gen_paths[i],
            "nearest_neighbor_idx": int(nn_idx),
            "nearest_neighbor_path": nn_path,
            "nearest_neighbor_label": nn_label,
            "memorization_ratio": float(ratio),
            "is_memorized": bool(sample_is_memorized),
        })
    
    # Compile results
    results = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "eval_step": args.eval_step,
        "num_samples": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "seed": args.seed,
        "metrics": {
            "f_mem": float(f_mem),
            "f_mem_percent": float(f_mem * 100),
            "memorization_threshold_k": args.mem_threshold,
            "num_memorized": len(memorized_indices),
            "num_non_memorized": len(non_memorized_indices),
        },
        "nearest_neighbor_info": nn_info,
    }
    
    # Save results
    results_path = os.path.join(eval_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Results saved to {results_path} ===")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Run: {args.run_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Memorization (f_mem): {f_mem*100:.2f}%")
    print(f"Memorized samples: {results['metrics']['num_memorized']}/{args.num_samples}")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Evaluation Script")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the run folder in outputs/cifar10/")
    parser.add_argument("--outputs-dir", type=str,
                        default="outputs/cifar10",
                        help="Path to outputs directory")
    parser.add_argument("--checkpoint", type=str, default="last.ckpt",
                        help="Checkpoint filename to load")
    parser.add_argument("--eval-step", type=str, default="final",
                        help="Step identifier for naming eval output folder")
    parser.add_argument("--cifar10-path", type=str, required=True,
                        help="Path to CIFAR-10 dataset root")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for generation")
    parser.add_argument("--sampling-steps", type=int, default=128,
                        help="Number of sampling steps")
    parser.add_argument("--mem-threshold", type=float, default=1/3,
                        help="Threshold k for memorization detection")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    main(args)
