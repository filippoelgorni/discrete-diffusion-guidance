"""Generate CIFAR-10 samples from trained model and save to disk.

This script loads a trained model and generates samples, saving them as PNG images.
Run this on GPU cluster, then download images to compute metrics locally.

Usage:
    python generate_cifar10_samples.py \
        --run-name mdlm_v1 \
        --checkpoint last.ckpt \
        --output-dir outputs/cifar10/mdlm_v1/generated_samples \
        --num-samples 10000 \
        --batch-size 64 \
        --seed 42
"""

import argparse
import json
import os
import typing
from datetime import datetime

import lightning as L
import numpy as np
import omegaconf
import torch
from PIL import Image
from tqdm import tqdm

import dataloader
import diffusion


def get_device():
    """Auto-detect available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def generate_samples(
    model: diffusion.Diffusion,
    num_samples: int,
    batch_size: int = 64,
    num_classes: int = 10,
) -> typing.Tuple[torch.Tensor, typing.List[int]]:
    """Generate samples from the diffusion model.
    
    Returns:
        samples: Generated images (N, 3, 32, 32) in range [0, 255]
        conditions: List of class conditions used for each sample (if CFG enabled)
    """
    model.eval()
    samples = []
    conditions = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Check if guidance is enabled
    use_guidance = (hasattr(model.config, 'guidance') 
                    and model.config.guidance is not None 
                    and model.config.guidance.method == 'cfg')
    
    print(f"Generating {num_samples} samples in {num_batches} batches...")
    print(f"Using CFG guidance: {use_guidance}")
    if use_guidance:
        print(f"  Condition: {model.config.guidance.condition}")
        print(f"  Gamma: {model.config.guidance.gamma}")
    
    for i in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - len(samples))
        model.config.sampling.batch_size = current_batch_size
        
        # Randomly sample condition for this batch if using guidance
        if use_guidance:
            batch_condition = np.random.randint(0, num_classes)
            model.config.guidance.condition = batch_condition
            conditions.extend([batch_condition] * current_batch_size)
        
        with torch.no_grad():
            raw_sample = model.sample()
            decoded = model.tokenizer.batch_decode(raw_sample).float()
            decoded = torch.clamp(decoded, 0, 255)
            samples.append(decoded)
    
    all_samples = torch.cat(samples, dim=0)[:num_samples]
    conditions = conditions[:num_samples] if use_guidance else [None] * num_samples
    
    assert all_samples.shape == (num_samples, 3, 32, 32), \
        f"Generated samples shape {all_samples.shape} != ({num_samples}, 3, 32, 32)"
    
    return all_samples, conditions


def save_images(
    images: torch.Tensor,
    save_dir: str,
    prefix: str = "sample",
    labels: typing.Optional[typing.List[typing.Optional[int]]] = None,
    class_names: typing.Optional[typing.List[str]] = None,
    use_class_subfolders: bool = True,
) -> typing.List[str]:
    """Save images to disk.
    
    Args:
        images: Tensor of images (N, C, H, W) in range [0, 255]
        save_dir: Directory to save images
        prefix: Prefix for filenames
        labels: Optional list of labels or conditions for each image
        class_names: Optional class name list for subfolder naming
        use_class_subfolders: If True, save into per-class subfolders
    
    Returns:
        List of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    
    for i, img in enumerate(tqdm(images, desc=f"Saving {prefix} images")):
        img_uint8 = torch.clamp(img.cpu(), 0, 255).to(torch.uint8)
        img_np = img_uint8.permute(1, 2, 0).numpy()
        
        pil_img = Image.fromarray(img_np)
        label = None
        if labels is not None:
            label = labels[i]

        if use_class_subfolders:
            if label is None:
                class_dir = "unconditional"
            else:
                if class_names is not None and 0 <= label < len(class_names):
                    class_dir = f"{label}_{class_names[label]}"
                else:
                    class_dir = str(label)
            class_dir_path = os.path.join(save_dir, class_dir)
            os.makedirs(class_dir_path, exist_ok=True)
            path = os.path.join(class_dir_path, f"{prefix}_{i:05d}.png")
        else:
            path = os.path.join(save_dir, f"{prefix}_{i:05d}.png")
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
    
    # Output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(
        checkpoint_dir, f"generated_samples_{args.eval_step}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Saving results to: {output_dir}")
    
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Log checkpoint file info
    ckpt_size = os.path.getsize(checkpoint_path)
    ckpt_mtime = os.path.getmtime(checkpoint_path)
    ckpt_time = datetime.fromtimestamp(ckpt_mtime).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Checkpoint file: {ckpt_size} bytes, modified: {ckpt_time}")
    
    # Load config
    config = omegaconf.OmegaConf.load(config_path)
    
    # Override sampling settings
    if args.sampling_steps is not None:
        config.sampling.steps = args.sampling_steps
    if args.batch_size is not None:
        config.sampling.batch_size = args.batch_size
    
    # Configure guidance
    if args.use_cfg:
        print(f"\n=== Configuring CFG guidance ===")
        print(f"  Method: cfg")
        print(f"  Condition: {args.cfg_condition}")
        print(f"  Gamma: {args.cfg_gamma}")
        guidance_config = {
            'method': 'cfg',
            'condition': args.cfg_condition,
            'gamma': args.cfg_gamma
        }
        omegaconf.OmegaConf.update(config, key='guidance', value=guidance_config, force_add=True)
    
    # Ensure eval settings are properly configured
    if not hasattr(config, 'eval'):
        config.eval = omegaconf.DictConfig({})
    if not hasattr(config.eval, 'disable_ema'):
        config.eval.disable_ema = False
    
    print(f"EMA disabled: {config.eval.disable_ema}")
    
    # Load tokenizer and model
    print("\n=== Loading model ===")
    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False,
    )
    device = get_device()
    print(f"Using device: {device}")
    try:
        model = model.to(device)
    except (RuntimeError, TypeError) as e:
        if 'mps' in str(e).lower() or 'float64' in str(e).lower():
            print(f"Device compatibility issue: {e}")
            print("Falling back to CPU...")
            model = model.to('cpu')
            device = torch.device('cpu')
        else:
            raise
    model.eval()
    
    # Log model state
    print(f"Global step (from checkpoint): {model.global_step}")
    if hasattr(model, 'current_epoch'):
        print(f"Epoch (from checkpoint): {model.current_epoch}")
    
    print(f"\n=== Model config ===")
    if hasattr(model.config, 'sampling'):
        print(f"Sampling config: steps={model.config.sampling.steps}, "
              f"batch_size={model.config.sampling.batch_size}")
    
    # Generate samples
    print(f"\n=== Generating {args.num_samples} samples ===")
    generated_samples, sample_conditions = generate_samples(
        model, args.num_samples, batch_size=model.config.sampling.batch_size,
        num_classes=config.data.num_classes)
    print(f"Generated {len(generated_samples)} samples")
    
    # Print condition distribution if using guidance
    if sample_conditions[0] is not None:
        condition_counts = {}
        for cond in sample_conditions:
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
        print(f"Condition distribution: {condition_counts}")
    
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    # Save generated images
    print("\n=== Saving generated images ===")
    gen_paths = save_images(
        generated_samples,
        output_dir,
        prefix="gen",
        labels=sample_conditions,
        class_names=class_names,
        use_class_subfolders=True,
    )
    
    # Save metadata
    metadata = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "eval_step": args.eval_step,
        "num_samples": args.num_samples,
        "sampling_steps": int(model.config.sampling.steps),
        "batch_size": int(model.config.sampling.batch_size),
        "seed": args.seed,
        "use_cfg": args.use_cfg,
        "cfg_condition": args.cfg_condition if args.use_cfg else None,
        "cfg_gamma": args.cfg_gamma if args.use_cfg else None,
        "generation_timestamp": datetime.now().isoformat(),
        "sample_info": [
            {
                "idx": i,
                "path": os.path.relpath(path, output_dir),
                "condition": sample_conditions[i]
            }
            for i, path in enumerate(gen_paths)
        ]
    }
    
    metadata_path = os.path.join(output_dir, "generation_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Run: {args.run_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 samples from trained model")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name of the run folder in outputs/cifar10/")
    parser.add_argument("--outputs-dir", type=str, default="outputs/cifar10",
                        help="Path to outputs directory")
    parser.add_argument("--checkpoint", type=str, default="last.ckpt",
                        help="Checkpoint filename to load")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save generated images (default: outputs/cifar10/<run_name>/generated_samples_<eval_step>)")
    parser.add_argument("--eval-step", type=str, default="final",
                        help="Step identifier for naming output folder")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for generation (uses config default if not specified)")
    parser.add_argument("--sampling-steps", type=int, default=None,
                        help="Number of sampling steps (uses config default if not specified)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use-cfg", action="store_true",
                        help="Enable classifier-free guidance (CFG)")
    parser.add_argument("--cfg-condition", type=int, default=0,
                        help="Class condition for CFG (0-9 for CIFAR-10)")
    parser.add_argument("--cfg-gamma", type=float, default=1.0,
                        help="Guidance strength for CFG (0=unconditional, 1=conditional)")
    
    args = parser.parse_args()
    main(args)
