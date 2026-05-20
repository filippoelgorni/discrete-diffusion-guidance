"""Reconstruct partially masked images using trained models.

This script loads images (from CIFAR-10 dataset or custom PNG files),
masks a portion (default: bottom half), and reconstructs them using
one or more trained checkpoints.

Usage:
    # Reconstruct a specific CIFAR-10 image
    python reconstruct_cifar10_images.py \
        --checkpoints outputs/cifar10/run1/checkpoints/last.ckpt \
        --index 42 \
        --mask-percentage 50 \
        --output-dir outputs/cifar10/reconstructions

    # Reconstruct random image from specific CIFAR-10 class
    python reconstruct_cifar10_images.py \
        --checkpoints outputs/cifar10/run1/checkpoints/last.ckpt \
        --category 3 \
        --mask-percentage 50

    # Reconstruct custom PNG image
    python reconstruct_cifar10_images.py \
        --checkpoints outputs/cifar10/run1/checkpoints/last.ckpt \
        --image-path path/to/image.png \
        --image-label 5 \
        --mask-percentage 50

    # Multiple checkpoints
    python reconstruct_cifar10_images.py \
        --checkpoints \
            outputs/cifar10/run1/checkpoints/last.ckpt \
            outputs/cifar10/run2/checkpoints/last.ckpt \
        --index 100 \
        --mask-percentage 50
"""

import argparse
import os
import typing
from datetime import datetime

import lightning as L
import numpy as np
import omegaconf
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

import dataloader
import diffusion


def load_cifar10_image(
    index: typing.Optional[int] = None,
    category: typing.Optional[int] = None,
    data_dir: str = "data/cifar10",
) -> typing.Tuple[torch.Tensor, int]:
    """Load a CIFAR-10 image.
    
    Args:
        index: Specific index (0-49999) to load. If None, uses category or random.
        category: If index is None, load random image from this category (0-9).
        data_dir: Directory containing CIFAR-10 data.
    
    Returns:
        image: Image tensor (3, 32, 32) in range [0, 255]
        label: Class label (0-9)
    """
    # Load full CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )
    
    if index is not None:
        # Load specific index
        if not 0 <= index < len(dataset):
            raise ValueError(f"Index {index} out of range [0, {len(dataset)})")
        img, label = dataset[index]
    elif category is not None:
        # Load random image from category
        if not 0 <= category <= 9:
            raise ValueError(f"Category {category} must be in range [0, 9]")
        # Find all images in this category
        indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == category]
        if not indices:
            raise ValueError(f"No images found for category {category}")
        index = np.random.choice(indices)
        img, label = dataset[index]
    else:
        # Random image
        index = np.random.randint(0, len(dataset))
        img, label = dataset[index]
    
    # Convert PIL to tensor (3, 32, 32) in range [0, 255]
    img_array = np.array(img)  # (32, 32, 3)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # (3, 32, 32)
    
    print(f"Loaded CIFAR-10 image: index={index}, label={label} ({get_class_name(label)})")
    return img_tensor, label


def load_image_from_path(
    image_path: str,
    label: typing.Optional[int] = None,
) -> typing.Tuple[torch.Tensor, typing.Optional[int]]:
    """Load an image from a PNG file path.
    
    Args:
        image_path: Path to PNG image file
        label: Optional class label (0-9 for CIFAR-10, or any int)
    
    Returns:
        image: Image tensor (3, H, W) in range [0, 255]
        label: Class label if provided, None otherwise
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded or doesn't have 3 channels
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load image with PIL
    img = Image.open(image_path)
    
    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to tensor (3, H, W) in range [0, 255]
    img_array = np.array(img)  # (H, W, 3)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()  # (3, H, W)
    
    label_str = f", label={label}" if label is not None else ""
    print(f"Loaded image from: {image_path}")
    print(f"  Shape: {img_tensor.shape}, dtype: {img_tensor.dtype}, range: [{img_tensor.min():.1f}, {img_tensor.max():.1f}]{label_str}")
    
    return img_tensor, label


def create_partial_mask(
    image: torch.Tensor,
    mask_percentage: float,
    mask_from_bottom: bool = True,
) -> torch.Tensor:
    """Create a partially masked image tensor suitable for reconstruction.
    
    Args:
        image: Image tensor (3, 32, 32) in range [0, 255]
        mask_percentage: Percentage of image to mask (0-100)
        mask_from_bottom: If True, mask bottom portion; if False, mask top portion
    
    Returns:
        masked_image: Tensor (3, 32, 32) with masked region set to black (0)
    """
    if not 0 <= mask_percentage <= 100:
        raise ValueError(f"mask_percentage must be in [0, 100], got {mask_percentage}")
    
    masked_image = image.clone()
    _, height, width = image.shape
    
    # Calculate number of rows to mask
    num_rows_to_mask = int(height * mask_percentage / 100)
    
    if mask_from_bottom:
        # Mask bottom rows
        masked_image[:, height - num_rows_to_mask:, :] = 0
    else:
        # Mask top rows
        masked_image[:, :num_rows_to_mask, :] = 0
    
    return masked_image


def create_random_mask(
    image: torch.Tensor,
    mask_percentage: float,
) -> torch.Tensor:
    """Create a randomly masked image tensor suitable for reconstruction.
    
    Masks pixels at random positions, but maintains the same mask pattern
    across all channels (i.e., if pixel (h, w) is masked, it's masked for
    all 3 color channels).
    
    Args:
        image: Image tensor (3, 32, 32) in range [0, 255]
        mask_percentage: Percentage of image pixels to mask (0-100)
    
    Returns:
        masked_image: Tensor (3, 32, 32) with randomly masked pixels set to black (0)
    """
    if not 0 <= mask_percentage <= 100:
        raise ValueError(f"mask_percentage must be in [0, 100], got {mask_percentage}")
    
    masked_image = image.clone()
    _, height, width = image.shape
    
    # Calculate total number of pixels to mask
    total_pixels = height * width
    num_pixels_to_mask = int(total_pixels * mask_percentage / 100)
    
    # Create random mask for spatial positions (H, W)
    spatial_mask = torch.rand(height, width) < (mask_percentage / 100)
    
    # Apply same mask across all channels
    masked_image[:, spatial_mask] = 0
    
    return masked_image


def encode_image_for_reconstruction(
    image: torch.Tensor,
    mask_percentage: float,
    tokenizer,
    mask_type: str = 'partial',
    mask_from_bottom: bool = True,
) -> torch.Tensor:
    """Encode image into token space with masking for reconstruction.
    
    Args:
        image: Image tensor (3, 32, 32) in range [0, 255]
        mask_percentage: Percentage of image to mask (0-100)
        tokenizer: Tokenizer for encoding
        mask_type: Type of masking - 'partial' (bottom/top half) or 'random' (random pixels)
        mask_from_bottom: If True, mask bottom portion; if False, mask top portion (only for 'partial')
        debug: If True, print detailed token statistics
    
    Returns:
        partial_tokens: Encoded tensor with some positions set to mask_index
    """
    # Encode full image to tokens
    image_batch = image.unsqueeze(0)  # (1, 3, 32, 32)
    tokens = tokenizer.batch_encode(image_batch)  # (1, L) where L = sequence length
    
    # Reshape tokens to (1, C, H, W) to apply SPATIAL masking (not token-sequential masking)
    batch_size, seq_len = tokens.shape
    tokens_3d = tokens.view(batch_size, 3, 32, 32)  # (1, 3, 32, 32)
    
    # Apply mask spatially (same mask across all channels)
    partial_tokens_3d = tokens_3d.clone()
    
    if mask_type == 'partial':
        # Calculate which rows to mask
        height = 32
        num_rows_to_mask = int(height * mask_percentage / 100)
        
        if mask_from_bottom:
            # Mask bottom rows across ALL channels
            partial_tokens_3d[:, :, height - num_rows_to_mask:, :] = tokenizer.mask_token_id
        else:
            # Mask top rows across ALL channels
            partial_tokens_3d[:, :, :num_rows_to_mask, :] = tokenizer.mask_token_id
    elif mask_type == 'random':
        # Create random mask
        height, width = 32, 32
        spatial_mask = torch.rand(height, width) < (mask_percentage / 100)
        # Apply same mask across all channels
        partial_tokens_3d[:, :, spatial_mask] = tokenizer.mask_token_id
    else:
        raise ValueError(f"mask_type must be 'partial' or 'random', got {mask_type}")
    
    # Flatten back to sequence
    partial_tokens = partial_tokens_3d.view(batch_size, seq_len)
    
    return partial_tokens


def reconstruct_image(
    model: diffusion.Diffusion,
    partial_tokens: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reconstruct image from partial tokens.
    
    Args:
        model: Diffusion model
        partial_tokens: Partially masked tokens (1, L)
        eps: Noise schedule epsilon
    
    Returns:
        reconstructed: Image tensor (3, 32, 32) in range [0, 255]
    """
    model.eval()
    with torch.no_grad():
        # Reconstruct
        reconstructed_tokens = model.reconstruct(partial_tokens, eps=eps)
        
        # Decode to image
        reconstructed_image = model.tokenizer.batch_decode(reconstructed_tokens).float()
        reconstructed_image = torch.clamp(reconstructed_image, 0, 255)
        
    return reconstructed_image.squeeze(0)  # (3, 32, 32)


def save_image(image: torch.Tensor, path: str):
    """Save image tensor to disk.
    
    Args:
        image: Image tensor (3, 32, 32) in range [0, 255]
        path: Output path
    """
    img_uint8 = torch.clamp(image, 0, 255).to(torch.uint8)
    img_np = img_uint8.permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(img_np)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_img.save(path)


def get_class_name(label: int) -> str:
    """Get CIFAR-10 class name."""
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    return class_names[label]


def load_model(checkpoint_path: str, device: str = 'cuda') -> diffusion.Diffusion:
    """Load model from checkpoint."""
    # Get config path
    checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(checkpoint_dir, ".hydra", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load config
    config = omegaconf.OmegaConf.load(config_path)
    
    # Ensure eval settings
    if not hasattr(config, 'eval'):
        config.eval = omegaconf.DictConfig({})
    if not hasattr(config.eval, 'disable_ema'):
        config.eval.disable_ema = False
    
    # Load tokenizer and model
    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config,
        strict=False,
    )
    model = model.to(device)
    model.eval()
    
    return model


def main(args):
    # Reproducibility
    L.seed_everything(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir if args.output_dir else f"outputs/cifar10/reconstructions/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("CIFAR-10 IMAGE RECONSTRUCTION")
    print("=" * 80)
    
    # Load image
    print("\n=== Loading image ===")
    if args.image_path:
        original_image, label = load_image_from_path(
            image_path=args.image_path,
            label=args.image_label,
        )
    else:
        original_image, label = load_cifar10_image(
            index=args.index,
            category=args.category,
            data_dir=args.data_dir,
        )
    
    # Determine if CFG should be used (enabled if label is provided)
    use_cfg = label is not None
    
    # Display guidance settings
    if use_cfg:
        print(f"\nCFG Guidance: ENABLED (label provided)")
        print(f"  Gamma (guidance strength): {args.cfg_gamma}")
        print(f"  Condition will be set to original image's class")
    else:
        print("\nCFG Guidance: DISABLED (no label provided)")
    
    # Save original image
    original_path = os.path.join(output_dir, "00_original.png")
    save_image(original_image, original_path)
    print(f"Original image saved: {original_path}")
    
    # We'll create the masked visualization after encoding to ensure consistency
    # (so the visualization matches exactly what the model sees)
    
    # Process each checkpoint
    print(f"\n=== Processing {len(args.checkpoints)} checkpoint(s) ===")
    
    for i, checkpoint_path in enumerate(args.checkpoints):
        print(f"\n--- Checkpoint {i + 1}/{len(args.checkpoints)} ---")
        print(f"Path: {checkpoint_path}")

        # Load model with guidance (use original image's label as condition if available)
        model, actual_device = load_model(
            checkpoint_path,
            device=args.device,
            use_cfg=use_cfg,
            cfg_condition=label,  # Use the original image's class as guidance
            cfg_gamma=args.cfg_gamma,
            sampling_steps=args.sampling_steps,
        )

        if use_cfg and label is not None:
            class_name = get_class_name(label) if label < 10 else "custom"
            print(f"Using CFG guidance with class {label} ({class_name}), gamma={args.cfg_gamma}")

        # Encode image with masking once (tokens are deterministic given image + tokenizer)
        print(f"Encoding image with {args.mask_percentage}% {args.mask_type} masking...")
        partial_tokens = encode_image_for_reconstruction(
            original_image,
            mask_percentage=args.mask_percentage,
            tokenizer=model.tokenizer,
            mask_type=args.mask_type,
            mask_from_bottom=args.mask_from_bottom,
        )
        partial_tokens = partial_tokens.to(actual_device)

        print(f"Token shape: {partial_tokens.shape}")
        num_masked = (partial_tokens == model.tokenizer.mask_token_id).sum().item()
        print(f"Number of masked tokens: {num_masked}/{partial_tokens.numel()}")

        # Run multiple reconstructions per checkpoint (different seeds)
        for run_idx in range(args.num_runs):
            run_seed = args.seed + run_idx
            print(f"\nRun {run_idx + 1}/{args.num_runs} (seed={run_seed})")
            L.seed_everything(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            # Reconstruct
            print(f"Reconstructing (using {model.config.sampling.steps} sampling steps)...")
            reconstructed_image = reconstruct_image(
                model,
                partial_tokens,
                eps=args.eps,
                mask_token_fill=args.mask_token_fill,
                log_token_stats=args.debug_tokens,
            )

            if args.debug_tokens:
                height = original_image.shape[1]
                num_rows_to_mask = int(height * args.mask_percentage / 100)
                if args.mask_from_bottom:
                    masked_rows = slice(height - num_rows_to_mask, height)
                else:
                    masked_rows = slice(0, num_rows_to_mask)
                mask = torch.zeros_like(original_image, dtype=torch.bool)
                mask[:, masked_rows, :] = True
                unmask = ~mask
                print("Region stats (original image)")
                _region_stats("  masked region", original_image, mask)
                _region_stats("  unmasked region", original_image, unmask)
                print("Region stats (reconstructed image)")
                _region_stats("  masked region", reconstructed_image, mask)
                _region_stats("  unmasked region", reconstructed_image, unmask)

            # Create masked visualization from actual tokens (only once, for first checkpoint/run)
            if i == 0 and run_idx == 0:
                print(f"\n=== Creating masked visualization from actual tokens ===")
                partial_tokens_vis = partial_tokens.clone()
                partial_tokens_vis[partial_tokens_vis == model.tokenizer.mask_token_id] = 0
                masked_visualization = model.tokenizer.batch_decode(partial_tokens_vis).float()
                masked_visualization = torch.clamp(masked_visualization, 0, 255).squeeze(0)

                masked_vis_path = os.path.join(output_dir, "01_masked.png")
                save_image(masked_visualization, masked_vis_path)
                print(f"Masked visualization saved: {masked_vis_path}")
                print(f"  (Generated from actual masked tokens to ensure consistency)")

            # Save reconstructed image (include run and seed)
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            run_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))
            recon_path = os.path.join(
                output_dir,
                f"02_reconstructed_{run_name}_{checkpoint_name}_run{run_idx}_seed{run_seed}.png"
            )
            save_image(reconstructed_image, recon_path)
            print(f"Reconstructed image saved: {recon_path}")

        # Clean up model after all runs for this checkpoint
        del model
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 80)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"Original image: {original_path}")
    if label is not None:
        class_name = get_class_name(label) if label < 10 else "custom"
        print(f"  Class: {label} ({class_name})")
    else:
        print(f"  Class: not specified")
    print(f"Masked visualization: {os.path.join(output_dir, '01_masked.png')}")
    print(f"  Mask type: {args.mask_type}")
    print(f"  Mask percentage: {args.mask_percentage}%")
    if args.mask_type == 'partial':
        print(f"  Mask direction: {'bottom' if args.mask_from_bottom else 'top'}")
    if use_cfg:
        print(f"CFG Guidance: ENABLED")
        if label is not None:
            class_name = get_class_name(label) if label < 10 else "custom"
            print(f"  Condition: {label} ({class_name})")
        print(f"  Gamma: {args.cfg_gamma}")
    else:
        print(f"CFG Guidance: DISABLED")
    print(f"Output directory: {output_dir}")
    total_recons = len(args.checkpoints) * getattr(args, 'num_runs', 1)
    print(f"Total reconstructions: {total_recons}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct partially masked CIFAR-10 images"
    )
    
    # Input selection
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to PNG image file. If provided, overrides CIFAR-10 options (--index, --category).",
    )
    parser.add_argument(
        "--image-label",
        type=int,
        default=None,
        help="Class label for custom image (optional, 0-9 for CIFAR-10 compatibility).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Specific CIFAR-10 image index (0-49999). Takes precedence over --category. Ignored if --image-path is provided.",
    )
    parser.add_argument(
        "--category",
        type=int,
        default=None,
        help="CIFAR-10 category (0-9). Loads random image from this category if --index not provided. Ignored if --image-path is provided.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/cifar10",
        help="Directory for CIFAR-10 data (ignored if --image-path is provided)",
    )
    
    # Masking
    parser.add_argument(
        "--mask-percentage",
        type=float,
        default=50.0,
        help="Percentage of image to mask (0-100)",
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="partial",
        choices=["partial", "random"],
        help="Type of masking: 'partial' (bottom/top region) or 'random' (random pixels)",
    )
    parser.add_argument(
        "--mask-from-bottom",
        action="store_true",
        default=True,
        help="Mask from bottom (default). Use --no-mask-from-bottom for top. Only applies to 'partial' mask type.",
    )
    parser.add_argument(
        "--no-mask-from-bottom",
        dest="mask_from_bottom",
        action="store_false",
        help="Mask from top instead of bottom. Only applies to 'partial' mask type.",
    )
    
    # Checkpoints
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to checkpoint file(s)",
    )
    
    # Diffusion parameters
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Noise schedule epsilon",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/cifar10/reconstructions/<timestamp>)",
    )
    
    # Guidance
    parser.add_argument(
        "--cfg-gamma",
        type=float,
        default=1.0,
        help="Guidance strength for CFG (0=unconditional, 1=conditional, >1=stronger guidance). CFG is enabled automatically when a label is provided.",
    )
    parser.add_argument(
        "--num-runs",
        "--num-samples-per-checkpoint",
        dest="num_runs",
        type=int,
        default=1,
        help="Number of reconstruction samples to generate per checkpoint (each uses a different seed).",
    )
    
    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.image_path is None:
        # Validating CIFAR-10 options only if not using custom image
        if args.index is not None and not 0 <= args.index < 50000:
            parser.error("--index must be in range [0, 49999]")
        if args.category is not None and not 0 <= args.category <= 9:
            parser.error("--category must be in range [0, 9]")
    if args.image_label is not None and not (isinstance(args.image_label, int)):
        parser.error("--image-label must be an integer")
    if not 0 <= args.mask_percentage <= 100:
        parser.error("--mask-percentage must be in range [0, 100]")
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if args.mask_token_fill is not None and not 0 <= args.mask_token_fill <= 255:
        parser.error("--mask-token-fill must be in range [0, 255]")
    
    # Verify checkpoints exist
    for ckpt in args.checkpoints:
        if not os.path.exists(ckpt):
            parser.error(f"Checkpoint not found: {ckpt}")
    
    main(args)
