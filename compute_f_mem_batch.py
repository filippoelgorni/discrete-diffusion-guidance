"""Batch compute f_mem for multiple models.

Usage:
    python compute_f_mem_batch.py models_config.txt --output results.csv --num-samples 10000

    # Optional CFG mode that draws a random class for every generated sample
    python compute_f_mem_batch.py models_config.txt --output results.csv --num-samples 10000 \
        --use-cfg-random-category --cfg-gamma 2.0

    models_config.txt format (pipe-separated):
        checkpoint_path | hydra_config_path | reference_dir
        outputs/model1/checkpoints/last.ckpt | outputs/model1/.hydra | data/cifar10_reference/train
        outputs/model2/checkpoints/best.ckpt | outputs/model2/.hydra | data/cifar10_reference/train
"""

import argparse
import csv
import os
from pathlib import Path

import lightning as L
import omegaconf
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm

from compute_cifar10_metrics import compute_memorization
from generate_cifar10_samples import generate_samples

import dataloader
import diffusion


HAMMING_THRESHOLDS_PCT_DEFAULT = [10.0, 5.0, 1.0, 0.5, 0.1]


def _ham_threshold_suffix(threshold_pct: float) -> str:
    """Format a Hamming threshold in percent as a compact CSV suffix."""
    return f"{int(round(threshold_pct * 10)):03d}"


def compute_hamming_memorization(
    generated: torch.Tensor,
    reference: torch.Tensor,
    thresholds_pct: list[float],
    chunk_size: int = 256,
) -> dict[str, float]:
    """Compute Hamming-based memorization fractions from one nearest-neighbor pass."""
    if len(generated.shape) != 4 or generated.shape[1:] != (3, 32, 32):
        raise ValueError(f"Generated samples shape {generated.shape} != (N, 3, 32, 32)")
    if len(reference.shape) != 4 or reference.shape[1:] != (3, 32, 32):
        raise ValueError(f"Reference images shape {reference.shape} != (M, 3, 32, 32)")

    thresholds_pct = sorted({float(t) for t in thresholds_pct})

    gen_flat = torch.clamp(generated, 0, 255).round().to(torch.uint8).view(generated.shape[0], -1).cpu().numpy()
    ref_flat = torch.clamp(reference, 0, 255).round().to(torch.uint8).view(reference.shape[0], -1).cpu().numpy()

    nearest_hamming = np.empty(len(gen_flat), dtype=np.float32)

    for i in tqdm(range(0, len(gen_flat), chunk_size), desc="  Hamming NN"):
        chunk = gen_flat[i:i + chunk_size]
        distances = cdist(chunk, ref_flat, metric="hamming")
        nearest_hamming[i:i + len(chunk)] = distances.min(axis=1)

    results = {}
    for threshold_pct in thresholds_pct:
        suffix = _ham_threshold_suffix(threshold_pct)
        results[f"f_mem_ham_{suffix}"] = float((nearest_hamming <= (threshold_pct / 100.0)).mean())

    return results


def load_images_from_dir(image_dir: str, label: str) -> torch.Tensor:
    """Load all PNG images from directory."""
    images = []
    img_paths = sorted(Path(image_dir).rglob('*.png'))
    for img_path in tqdm(img_paths, desc=f"  Loading {label} images", leave=False):
        pil_img = Image.open(img_path)
        img_array = __import__('numpy').array(pil_img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        images.append(img_tensor)
    
    images_tensor = torch.stack(images)
    print(f"  {label}: {len(images_tensor)} images")
    return images_tensor


def generate_samples_with_random_cfg_categories(
    model: diffusion.Diffusion,
    num_samples: int,
    num_classes: int,
) -> tuple[torch.Tensor, list[int]]:
    """Generate samples one at a time, picking a random CFG class per sample."""
    if not hasattr(model.config, 'guidance') or model.config.guidance is None:
        raise ValueError("CFG guidance must be configured before calling this helper")
    if getattr(model.config.guidance, 'method', None) != 'cfg':
        raise ValueError("model.config.guidance.method must be 'cfg' for random CFG sampling")

    samples = []
    conditions = []
    original_batch_size = int(model.config.sampling.batch_size)

    print(f"Generating {num_samples} samples one-by-one with random CFG categories...")
    print(f"Using CFG guidance: True")
    print(f"  Gamma: {model.config.guidance.gamma}")
    print(f"  Categories sampled uniformly from [0, {num_classes - 1}]")

    model.config.sampling.batch_size = 1
    try:
        for _ in tqdm(range(num_samples), desc="Generating samples"):
            batch_condition = int(np.random.randint(0, num_classes))
            model.config.guidance.condition = batch_condition
            conditions.append(batch_condition)

            with torch.no_grad():
                raw_sample = model.sample()
                decoded = model.tokenizer.batch_decode(raw_sample).float()
                decoded = torch.clamp(decoded, 0, 255)
                samples.append(decoded)
    finally:
        model.config.sampling.batch_size = original_batch_size

    all_samples = torch.cat(samples, dim=0)[:num_samples]
    assert all_samples.shape == (num_samples, 3, 32, 32), (
        f"Generated samples shape {all_samples.shape} != ({num_samples}, 3, 32, 32)"
    )

    return all_samples, conditions


def main(args):
    # Read model config
    models = []
    with open(args.config) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3:
                print(f"WARNING: Skipping malformed line: {line}")
                continue
            models.append({
                'checkpoint_path': parts[0],
                'hydra_config_path': parts[1],
                'reference_dir': parts[2]
            })
    
    print(f"Computing f_mem for {len(models)} models...\n")

    ham_thresholds_pct = list(args.ham_thresholds_pct)
    ham_fieldnames = [f"f_mem_ham_{_ham_threshold_suffix(t)}" for t in sorted({float(t) for t in ham_thresholds_pct})]
    
    # Open CSV file for writing (write header immediately)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    csv_file = open(args.output, 'w', newline='')
    writer = csv.DictWriter(
        csv_file,
        fieldnames=['model', 'checkpoint', 'reference_dir', 'f_mem', 'f_mem_percent', *ham_fieldnames, 'status'],
    )
    writer.writeheader()
    csv_file.flush()
    
    try:
        for model_idx, model in enumerate(tqdm(models, desc="Processing models", unit="model")):
            model_name = Path(model['checkpoint_path']).parent.parent.name
            print(f"\n[{model_idx + 1}/{len(models)}] {model_name}")
            print(f"  Checkpoint: {model['checkpoint_path']}")
            print(f"  Config: {model['hydra_config_path']}")
            print(f"  Reference: {model['reference_dir']}")
            
            try:
                # Load reference images for this model
                reference_images = load_images_from_dir(model['reference_dir'], "Reference")

                config_path = Path(model['hydra_config_path'])
                if config_path.is_dir():
                    config_path = config_path / "config.yaml"
                if not config_path.exists():
                    raise FileNotFoundError(f"Config not found: {config_path}")

                print(f"  Loading config: {config_path}")
                config = omegaconf.OmegaConf.load(str(config_path))

                if args.sampling_steps is not None:
                    config.sampling.steps = args.sampling_steps
                if args.batch_size is not None:
                    config.sampling.batch_size = args.batch_size

                if args.use_cfg_random_category:
                    guidance_config = {
                        'method': 'cfg',
                        'condition': 0,
                        'gamma': args.cfg_gamma,
                    }
                    omegaconf.OmegaConf.update(config, key='guidance', value=guidance_config, force_add=True)

                if not hasattr(config, 'eval'):
                    config.eval = omegaconf.DictConfig({})
                if not hasattr(config.eval, 'disable_ema'):
                    config.eval.disable_ema = False

                print(f"  Loading model checkpoint...")
                tokenizer = dataloader.get_tokenizer(config)
                model_obj = diffusion.Diffusion.load_from_checkpoint(
                    model['checkpoint_path'],
                    tokenizer=tokenizer,
                    config=config,
                    strict=False,
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_obj = model_obj.to(device)
                model_obj.eval()
                print(f"  Model loaded on {device}.")

                num_classes = 10
                if hasattr(config, "data") and hasattr(config.data, "num_classes"):
                    num_classes = int(config.data.num_classes)

                print(f"  Generating {args.num_samples} samples on {device}")
                if args.use_cfg_random_category:
                    generated_images, _ = generate_samples_with_random_cfg_categories(
                        model_obj,
                        args.num_samples,
                        num_classes=num_classes,
                    )
                else:
                    generated_images, _ = generate_samples(
                        model_obj,
                        args.num_samples,
                        batch_size=model_obj.config.sampling.batch_size,
                        num_classes=num_classes,
                    )
                generated_images = generated_images.cpu()

                print(f"  Computing f_mem metric...")
                f_mem, _, _ = compute_memorization(generated_images, reference_images, k=args.mem_threshold)
                print(f"  Computing Hamming f_mem metrics...")
                hamming_metrics = compute_hamming_memorization(
                    generated_images,
                    reference_images,
                    thresholds_pct=ham_thresholds_pct,
                )
                
                result = {
                    'model': model_name,
                    'checkpoint': model['checkpoint_path'],
                    'reference_dir': model['reference_dir'],
                    'f_mem': f_mem,
                    'f_mem_percent': f_mem * 100,
                    **hamming_metrics,
                    'status': 'ok'
                }
                writer.writerow(result)
                csv_file.flush()
                print(f"  f_mem: {f_mem*100:.2f}%")
                for key in ham_fieldnames:
                    print(f"  {key}: {hamming_metrics[key]*100:.2f}%")
            except Exception as e:
                result = {
                    'model': model_name,
                    'checkpoint': model['checkpoint_path'],
                    'reference_dir': model['reference_dir'],
                    'f_mem': None,
                    'f_mem_percent': None,
                    **{name: None for name in ham_fieldnames},
                    'status': f'error: {str(e)}'
                }
                writer.writerow(result)
                csv_file.flush()
                print(f"  ERROR: {e}")
    finally:
        csv_file.close()
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compute f_mem for multiple models")
    parser.add_argument("config", type=str, help="Config file (pipe-separated: checkpoint_path | hydra_config_path | reference_dir)")
    parser.add_argument("--output", type=str, default="f_mem_results.csv", help="Output CSV file")
    parser.add_argument("--mem-threshold", type=float, default=1/3, help="Memorization threshold k")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate per model")
    parser.add_argument(
        "--ham-thresholds-pct",
        type=float,
        nargs="+",
        default=HAMMING_THRESHOLDS_PCT_DEFAULT,
        help="Hamming thresholds in percent of differing pixels. Default: 10 5 1 0.5 0.1",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for generation (uses config default if not specified)")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Number of sampling steps (uses config default if not specified)")
    parser.add_argument(
        "--use-cfg-random-category",
        action="store_true",
        help="Enable CFG and sample a fresh random class for every generated sample.",
    )
    parser.add_argument(
        "--cfg-gamma",
        type=float,
        default=1.0,
        help="CFG guidance strength used when --use-cfg-random-category is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    L.seed_everything(args.seed)
    main(args)
