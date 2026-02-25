"""Batch compute f_mem for multiple models.

Usage:
    python compute_f_mem_batch.py models_config.txt --output results.csv --num-samples 10000

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
import torch
from PIL import Image
from tqdm import tqdm

from compute_cifar10_metrics import compute_memorization
from generate_cifar10_samples import generate_samples

import dataloader
import diffusion


def load_images_from_dir(image_dir: str, label: str) -> torch.Tensor:
    """Load all PNG images from directory."""
    images = []
    for img_path in sorted(Path(image_dir).rglob('*.png')):
        pil_img = Image.open(img_path)
        img_array = __import__('numpy').array(pil_img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        images.append(img_tensor)
    
    images_tensor = torch.stack(images)
    print(f"  {label}: {len(images_tensor)} images")
    return images_tensor


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
    
    # Open CSV file for writing (write header immediately)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    csv_file = open(args.output, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['model', 'checkpoint', 'reference_dir', 'f_mem', 'f_mem_percent', 'status'])
    writer.writeheader()
    csv_file.flush()
    
    try:
        for model_idx, model in enumerate(models):
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

                if not hasattr(config, 'eval'):
                    config.eval = omegaconf.DictConfig({})
                if not hasattr(config.eval, 'disable_ema'):
                    config.eval.disable_ema = False

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

                num_classes = 10
                if hasattr(config, "data") and hasattr(config.data, "num_classes"):
                    num_classes = int(config.data.num_classes)

                print(f"  Generating {args.num_samples} samples on {device}")
                generated_images, _ = generate_samples(
                    model_obj,
                    args.num_samples,
                    batch_size=model_obj.config.sampling.batch_size,
                    num_classes=num_classes,
                )
                generated_images = generated_images.cpu()

                f_mem, _, _ = compute_memorization(generated_images, reference_images, k=args.mem_threshold)
                
                result = {
                    'model': model_name,
                    'checkpoint': model['checkpoint_path'],
                    'reference_dir': model['reference_dir'],
                    'f_mem': f_mem,
                    'f_mem_percent': f_mem * 100,
                    'status': 'ok'
                }
                writer.writerow(result)
                csv_file.flush()
                print(f"  f_mem: {f_mem*100:.2f}%")
            except Exception as e:
                result = {
                    'model': model_name,
                    'checkpoint': model['checkpoint_path'],
                    'reference_dir': model['reference_dir'],
                    'f_mem': None,
                    'f_mem_percent': None,
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
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for generation (uses config default if not specified)")
    parser.add_argument("--sampling-steps", type=int, default=None, help="Number of sampling steps (uses config default if not specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    L.seed_everything(args.seed)
    main(args)
