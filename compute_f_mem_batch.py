"""Batch compute f_mem for multiple models.

Usage:
    python compute_f_mem_batch.py models_config.txt --output results.csv

    models_config.txt format (pipe-separated):
        checkpoint_path | hydra_config_path | reference_dir
        outputs/model1/checkpoints/last.ckpt | outputs/model1/.hydra | data/cifar10_reference/train
        outputs/model2/checkpoints/best.ckpt | outputs/model2/.hydra | data/cifar10_reference/train
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm


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


def compute_memorization(generated: torch.Tensor, reference: torch.Tensor, k: float = 1/3) -> float:
    """Compute f_mem."""
    gen_flat = generated.view(generated.shape[0], -1).numpy()
    ref_flat = reference.view(reference.shape[0], -1).numpy()
    
    mem_ratios = []
    for i in tqdm(range(0, len(gen_flat), 500), desc="  Computing distances", leave=False):
        chunk = gen_flat[i:i + 500]
        distances = cdist(chunk, ref_flat, metric='euclidean')
        sorted_idx = __import__('numpy').argsort(distances, axis=1)
        d1 = distances[__import__('numpy').arange(len(chunk)), sorted_idx[:, 0]]
        d2 = distances[__import__('numpy').arange(len(chunk)), sorted_idx[:, 1]]
        mem_ratios.extend((d1 / (d2 + 1e-8)).tolist())
    
    f_mem = sum(r < k for r in mem_ratios) / len(mem_ratios)
    return f_mem


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
    
    results = []
    
    for model_idx, model in enumerate(models):
        model_name = Path(model['checkpoint_path']).parent.parent.name
        print(f"\n[{model_idx + 1}/{len(models)}] {model_name}")
        print(f"  Checkpoint: {model['checkpoint_path']}")
        print(f"  Config: {model['hydra_config_path']}")
        print(f"  Reference: {model['reference_dir']}")
        
        try:
            # Load reference images for this model
            reference_images = load_images_from_dir(model['reference_dir'], "Reference")
            
            # Find generated samples (look for common naming patterns)
            gen_dir = model['checkpoint_path'].rsplit('/', 2)[0] + '/generated_samples*'
            gen_dirs = sorted(Path('.').glob(gen_dir.replace('./', '')))
            
            if not gen_dirs:
                # Fallback: look for outputs dir structure
                model_dir = Path(model['checkpoint_path']).parent.parent.parent
                gen_dirs = list(model_dir.glob('generated_samples*'))
            
            if not gen_dirs:
                raise FileNotFoundError(f"No generated_samples directory found for {model_name}")
            
            gen_dir = str(gen_dirs[-1])
            print(f"  Generated: {gen_dir}")
            
            generated_images = load_images_from_dir(gen_dir, "Generated")
            f_mem = compute_memorization(generated_images, reference_images, k=args.mem_threshold)
            
            results.append({
                'model': model_name,
                'checkpoint': model['checkpoint_path'],
                'reference_dir': model['reference_dir'],
                'f_mem': f_mem,
                'f_mem_percent': f_mem * 100,
                'status': 'ok'
            })
            print(f"  f_mem: {f_mem*100:.2f}%")
        except Exception as e:
            results.append({
                'model': model_name,
                'checkpoint': model['checkpoint_path'],
                'reference_dir': model['reference_dir'],
                'f_mem': None,
                'f_mem_percent': None,
                'status': f'error: {str(e)}'
            })
            print(f"  ERROR: {e}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'checkpoint', 'reference_dir', 'f_mem', 'f_mem_percent', 'status'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compute f_mem for multiple models")
    parser.add_argument("config", type=str, help="Config file (pipe-separated: checkpoint_path | hydra_config_path | reference_dir)")
    parser.add_argument("--output", type=str, default="f_mem_results.csv", help="Output CSV file")
    parser.add_argument("--mem-threshold", type=float, default=1/3, help="Memorization threshold k")
    
    args = parser.parse_args()
    main(args)
