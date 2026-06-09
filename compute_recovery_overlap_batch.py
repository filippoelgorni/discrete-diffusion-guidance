"""Batch compute recovery R(t) and overlap O(t) for CIFAR-10 reference reconstructions.

This script combines the batch model/config handling from compute_f_mem_batch.py
with the random masking + guided reconstruction flow from reconstruct_cifar10_images.py.

Config file format, pipe-separated:

    checkpoint_path | hydra_config_path | reference_dir

where reference_dir is expected to contain CIFAR-10 class subfolders such as:

    cifar10_1_reference/train/0_airplane
    cifar10_1_reference/train/1_automobile
    ...

For each model and each time t, the script draws n_samples random reference
images. Each draw chooses a class subfolder uniformly, infers the class label
from the folder prefix, chooses a random image from that folder, masks fraction t
of spatial positions, reconstructs with CFG guidance on the inferred class and
cfg_gamma, then computes:

    recovery R(t) = 1 - d_ham(t) / N_mask,    N_mask = t * 3072

where d_ham(t) is the Hamming-count distance between the generated full image
and the closest reference image in Hamming sense, over all 3072 RGB entries.
At t=0, R(t) is undefined because N_mask=0, so the CSV value is NaN.

    overlap O(t) = <x_mu, x(t)> / (||x_mu||_2 ||x(t)||_2)

where x_mu is the closest reference image in L2 sense.

The output is one aggregate CSV row per model and time t, with recovery and
overlap averaged over n_samples.

Example:

    python compute_recovery_overlap_batch.py f_mem_models_100000.txt \
        --output recovery_overlap_results.csv \
        --n-samples 1000 \
        --times 0.0:0.1:1.0 \
        --cfg-gamma 1.0
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import typing
from pathlib import Path

import lightning as L
import numpy as np
import omegaconf
import torch
from PIL import Image
from tqdm import tqdm

import dataloader
import diffusion


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TOKEN_DIM = 3 * 32 * 32
SPATIAL_PIXELS = 32 * 32


def parse_times(times_spec: str) -> list[float]:
    """Parse either comma-separated times or MATLAB-like start:step:end."""
    times_spec = times_spec.strip()
    if not times_spec:
        raise ValueError("--times cannot be empty")

    if ":" in times_spec:
        parts = [float(x) for x in times_spec.split(":")]
        if len(parts) != 3:
            raise ValueError("Colon format must be start:step:end, e.g. 0.0:0.1:1.0")
        start, step, end = parts
        if step <= 0:
            raise ValueError("Time step must be positive")
        values = []
        x = start
        while x <= end + 1e-12:
            values.append(round(float(x), 10))
            x += step
        return values

    values = [float(x.strip()) for x in times_spec.split(",") if x.strip()]
    if not values:
        raise ValueError("No valid times found")
    return values


def validate_times(times: list[float]) -> None:
    for t in times:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"All times/mask fractions must be in [0, 1], got {t}")


def parse_model_config(config_path: str) -> list[dict[str, str]]:
    models = []
    with open(config_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                print(f"WARNING: skipping malformed line {line_no}: {line}")
                continue
            models.append(
                {
                    "checkpoint_path": parts[0],
                    "hydra_config_path": parts[1],
                    "reference_dir": parts[-1],
                    "line_no": str(line_no),
                }
            )
    if not models:
        raise ValueError(f"No valid model lines found in {config_path}")
    return models


def infer_category_from_folder(folder: Path) -> int:
    """Infer class id from folders such as 0_airplane or 7_horse."""
    match = re.match(r"^(\d+)(?:[_-].*)?$", folder.name)
    if not match:
        raise ValueError(
            f"Cannot infer category from folder name {folder.name!r}; expected e.g. '0_airplane'"
        )
    category = int(match.group(1))
    if not 0 <= category <= 9:
        raise ValueError(f"Inferred category {category} from {folder}, expected 0..9")
    return category


def list_image_paths(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_reference_subfolders(reference_root: str) -> list[tuple[int, Path, list[Path]]]:
    root = Path(reference_root)
    if not root.exists():
        raise FileNotFoundError(f"Reference root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Reference root is not a directory: {root}")

    class_dirs: list[tuple[int, Path, list[Path]]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            category = infer_category_from_folder(child)
        except ValueError:
            continue
        image_paths = list_image_paths(child)
        if image_paths:
            class_dirs.append((category, child, image_paths))

    if not class_dirs:
        raise ValueError(
            f"No class subfolders with images found under {root}. Expected folders like 0_airplane."
        )
    return class_dirs


def load_image_from_path(image_path: Path) -> torch.Tensor:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.Resampling.BICUBIC)
    img_array = np.array(img)
    return torch.from_numpy(img_array).permute(2, 0, 1).float()


def load_reference_images(reference_root: str) -> tuple[torch.Tensor, list[Path], list[int], list[tuple[int, Path, list[Path]]]]:
    """Load all reference images and return tensor plus sampling metadata."""
    class_dirs = discover_reference_subfolders(reference_root)
    images: list[torch.Tensor] = []
    paths: list[Path] = []
    labels: list[int] = []

    for category, _folder, image_paths in class_dirs:
        for image_path in tqdm(image_paths, desc=f"  Loading ref class {category}", leave=False):
            images.append(load_image_from_path(image_path))
            paths.append(image_path)
            labels.append(category)

    if not images:
        raise ValueError(f"No reference images found under {reference_root}")

    reference_images = torch.stack(images, dim=0)
    print(f"  Reference images loaded: {len(reference_images)}")
    return reference_images, paths, labels, class_dirs


def encode_image_with_random_mask(
    image: torch.Tensor,
    mask_fraction: float,
    tokenizer,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode image and mask a fraction of spatial positions.

    Returns:
        partial_tokens: shape (1, 3072) for CIFAR-10 RGB 32x32 tokenizers.
        spatial_mask: bool tensor of shape (32, 32), True means masked.
    """
    if not 0.0 <= mask_fraction <= 1.0:
        raise ValueError(f"mask_fraction must be in [0, 1], got {mask_fraction}")

    image_batch = image.unsqueeze(0)
    tokens = tokenizer.batch_encode(image_batch)
    batch_size, seq_len = tokens.shape
    if seq_len != TOKEN_DIM:
        raise ValueError(f"Expected token sequence length {TOKEN_DIM}, got {seq_len}")

    tokens_3d = tokens.view(batch_size, 3, 32, 32).clone()
    n_mask = int(round(SPATIAL_PIXELS * mask_fraction))

    spatial_mask = torch.zeros(SPATIAL_PIXELS, dtype=torch.bool)
    if n_mask > 0:
        perm = torch.randperm(SPATIAL_PIXELS, generator=generator)
        spatial_mask[perm[:n_mask]] = True
    spatial_mask = spatial_mask.view(32, 32)

    tokens_3d[:, :, spatial_mask] = tokenizer.mask_token_id
    partial_tokens = tokens_3d.view(batch_size, seq_len)
    return partial_tokens, spatial_mask


def configure_eval_and_guidance(
    config,
    cfg_condition: int,
    cfg_gamma: float,
    sampling_steps: int,
    batch_size: int,
) -> None:
    config.sampling.steps = int(sampling_steps)
    config.sampling.batch_size = int(batch_size)

    guidance_config = {
        "method": "cfg",
        "condition": int(cfg_condition),
        "gamma": float(cfg_gamma),
    }
    omegaconf.OmegaConf.update(config, key="guidance", value=guidance_config, force_add=True)

    if not hasattr(config, "eval"):
        config.eval = omegaconf.DictConfig({})
    if not hasattr(config.eval, "disable_ema"):
        config.eval.disable_ema = False


def load_model_from_config_line(
    checkpoint_path: str,
    hydra_config_path: str,
    device: str,
    batch_size: int,
) -> diffusion.Diffusion:
    config_path = Path(hydra_config_path)
    if config_path.is_dir():
        config_path = config_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = omegaconf.OmegaConf.load(str(config_path))
    configure_eval_and_guidance(
        config,
        cfg_condition=0,
        cfg_gamma=1.0,
        sampling_steps=1,
        batch_size=batch_size,
    )

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


def sampling_steps_for_time(t: float, max_steps_base: int, reconstruct_t_zero: bool) -> int:
    steps = int(round(max_steps_base * t))
    if t <= 0.0 and not reconstruct_t_zero:
        return 0
    return max(1, steps)


def reconstruct_from_partial_tokens(
    model: diffusion.Diffusion,
    partial_tokens: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        reconstructed_tokens = model.reconstruct(partial_tokens, eps=eps)
        reconstructed_image = model.tokenizer.batch_decode(reconstructed_tokens).float()
        reconstructed_image = torch.clamp(reconstructed_image, 0, 255)
    return reconstructed_image.squeeze(0).cpu()


def safe_model_name(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    if path.parent and path.parent.parent:
        return path.parent.parent.name
    return path.stem


def nearest_hamming_count(generated: torch.Tensor, reference_flat_uint8: torch.Tensor, chunk_size: int) -> int:
    """Return min Hamming count over reference images."""
    gen_flat = torch.clamp(generated, 0, 255).round().to(torch.uint8).view(1, -1)
    best = TOKEN_DIM + 1
    for start in range(0, reference_flat_uint8.shape[0], chunk_size):
        ref_chunk = reference_flat_uint8[start:start + chunk_size]
        distances = (ref_chunk != gen_flat).sum(dim=1)
        chunk_best = int(distances.min().item())
        if chunk_best < best:
            best = chunk_best
            if best == 0:
                break
    return best


def nearest_l2_reference_and_overlap(
    generated: torch.Tensor,
    reference_flat_float: torch.Tensor,
    chunk_size: int,
) -> tuple[int, float, float]:
    """Return nearest L2 index, squared L2 distance, and cosine overlap."""
    gen_flat = torch.clamp(generated, 0, 255).float().view(-1)
    gen_norm = torch.linalg.vector_norm(gen_flat)

    best_idx = -1
    best_dist = float("inf")
    best_ref_flat: torch.Tensor | None = None

    for start in range(0, reference_flat_float.shape[0], chunk_size):
        ref_chunk = reference_flat_float[start:start + chunk_size]
        diff = ref_chunk - gen_flat.unsqueeze(0)
        distances = (diff * diff).sum(dim=1)
        chunk_dist, chunk_pos = torch.min(distances, dim=0)
        chunk_dist_float = float(chunk_dist.item())
        if chunk_dist_float < best_dist:
            best_dist = chunk_dist_float
            best_idx = start + int(chunk_pos.item())
            best_ref_flat = ref_chunk[int(chunk_pos.item())].clone()

    if best_ref_flat is None or best_idx < 0:
        raise RuntimeError("Failed to find nearest L2 reference image")

    ref_norm = torch.linalg.vector_norm(best_ref_flat)
    denom = float((ref_norm * gen_norm).item())
    if denom == 0.0:
        overlap = float("nan")
    else:
        overlap = float(torch.dot(best_ref_flat, gen_flat).item() / denom)

    return best_idx, best_dist, overlap


def compute_metrics_for_image(
    generated: torch.Tensor,
    t: float,
    reference_flat_uint8: torch.Tensor,
    reference_flat_float: torch.Tensor,
    hamming_chunk_size: int,
    l2_chunk_size: int,
) -> tuple[float, float, int, int, float]:
    """Compute recovery and overlap for one generated image.

    Returns recovery, overlap, hamming_count, l2_nearest_idx, l2_distance_sq.
    """
    d_ham = nearest_hamming_count(generated, reference_flat_uint8, hamming_chunk_size)
    n_mask = float(t) * TOKEN_DIM
    recovery = float("nan") if n_mask == 0.0 else 1.0 - (float(d_ham) / n_mask)

    l2_idx, l2_dist_sq, overlap = nearest_l2_reference_and_overlap(
        generated,
        reference_flat_float,
        l2_chunk_size,
    )
    return recovery, overlap, d_ham, l2_idx, l2_dist_sq


def write_error_rows(
    writer: csv.DictWriter,
    csv_file,
    model_idx: int,
    model_name: str,
    checkpoint_path: str,
    reference_root: str,
    times: list[float],
    error: Exception,
) -> None:
    for t in times:
        writer.writerow(
            {
                "model": model_name,
                "checkpoint": checkpoint_path,
                "reference_dir": reference_root,
                "t": t,
                "recovery": "",
                "overlap": "",
                "recovery_percent": "",
                "overlap_percent": "",
                "n_samples": "",
                "sampling_steps": "",
                "mean_hamming_count": "",
                "mean_l2_distance_sq": "",
                "model_idx": model_idx,
                "status": f"error: {error}",
            }
        )
    csv_file.flush()


def main(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    times = parse_times(args.times)
    validate_times(times)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    print(f"Times/mask fractions: {times}")

    models = parse_model_config(args.config)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fieldnames = [
        "model",
        "checkpoint",
        "reference_dir",
        "t",
        "recovery",
        "overlap",
        "recovery_percent",
        "overlap_percent",
        "n_samples",
        "sampling_steps",
        "mean_hamming_count",
        "mean_l2_distance_sq",
        "model_idx",
        "status",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

        for model_idx, model_spec in enumerate(tqdm(models, desc="Processing models", unit="model")):
            checkpoint_path = model_spec["checkpoint_path"]
            hydra_config_path = model_spec["hydra_config_path"]
            reference_root = model_spec["reference_dir"]
            model_name = safe_model_name(checkpoint_path)

            print(f"\n[{model_idx + 1}/{len(models)}] {model_name}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"  Config:     {hydra_config_path}")
            print(f"  Reference:  {reference_root}")

            try:
                reference_images, _ref_paths, _ref_labels, class_dirs = load_reference_images(reference_root)
                reference_flat_uint8 = torch.clamp(reference_images, 0, 255).round().to(torch.uint8).view(reference_images.shape[0], -1).cpu()
                reference_flat_float = torch.clamp(reference_images, 0, 255).float().view(reference_images.shape[0], -1).cpu()

                model_obj = load_model_from_config_line(
                    checkpoint_path=checkpoint_path,
                    hydra_config_path=hydra_config_path,
                    device=args.device,
                    batch_size=1,
                )

                rng = np.random.default_rng(args.seed + model_idx)
                mask_generator = torch.Generator(device="cpu")
                mask_generator.manual_seed(args.seed + 10_000 * (model_idx + 1))

                for t in tqdm(times, desc=f"Times {model_name}", leave=False):
                    steps = sampling_steps_for_time(
                        t,
                        max_steps_base=args.max_steps_base,
                        reconstruct_t_zero=args.reconstruct_t_zero,
                    )
                    recoveries: list[float] = []
                    overlaps: list[float] = []
                    hamming_counts: list[float] = []
                    l2_distances_sq: list[float] = []

                    for _sample_idx in tqdm(range(args.n_samples), desc=f"Samples t={t:g}", leave=False):
                        category, _class_dir, image_paths = class_dirs[int(rng.integers(0, len(class_dirs)))]
                        image_path = image_paths[int(rng.integers(0, len(image_paths)))]

                        model_obj.config.guidance.condition = int(category)
                        model_obj.config.guidance.gamma = float(args.cfg_gamma)
                        model_obj.config.sampling.steps = max(1, steps) if steps > 0 else 1

                        original = load_image_from_path(image_path)
                        partial_tokens, _ = encode_image_with_random_mask(
                            original,
                            mask_fraction=t,
                            tokenizer=model_obj.tokenizer,
                            generator=mask_generator,
                        )
                        partial_tokens = partial_tokens.to(args.device)

                        if steps == 0:
                            generated = original.cpu()
                        else:
                            generated = reconstruct_from_partial_tokens(
                                model_obj,
                                partial_tokens=partial_tokens,
                                eps=args.eps,
                            )

                        recovery, overlap, d_ham, _l2_idx, l2_dist_sq = compute_metrics_for_image(
                            generated,
                            t=t,
                            reference_flat_uint8=reference_flat_uint8,
                            reference_flat_float=reference_flat_float,
                            hamming_chunk_size=args.hamming_chunk_size,
                            l2_chunk_size=args.l2_chunk_size,
                        )
                        recoveries.append(recovery)
                        overlaps.append(overlap)
                        hamming_counts.append(float(d_ham))
                        l2_distances_sq.append(float(l2_dist_sq))

                    recovery_mean = float(np.nanmean(recoveries)) if recoveries else float("nan")
                    overlap_mean = float(np.nanmean(overlaps)) if overlaps else float("nan")
                    hamming_mean = float(np.nanmean(hamming_counts)) if hamming_counts else float("nan")
                    l2_mean = float(np.nanmean(l2_distances_sq)) if l2_distances_sq else float("nan")

                    result = {
                        "model": model_name,
                        "checkpoint": checkpoint_path,
                        "reference_dir": reference_root,
                        "t": t,
                        "recovery": recovery_mean,
                        "overlap": overlap_mean,
                        "recovery_percent": recovery_mean * 100.0 if not np.isnan(recovery_mean) else "",
                        "overlap_percent": overlap_mean * 100.0 if not np.isnan(overlap_mean) else "",
                        "n_samples": args.n_samples,
                        "sampling_steps": steps,
                        "mean_hamming_count": hamming_mean,
                        "mean_l2_distance_sq": l2_mean,
                        "model_idx": model_idx,
                        "status": "ok",
                    }
                    writer.writerow(result)
                    csv_file.flush()

                    print(
                        f"  t={t:g}: recovery={recovery_mean:.6g}, "
                        f"overlap={overlap_mean:.6g}, steps={steps}"
                    )

                del model_obj
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                write_error_rows(
                    writer=writer,
                    csv_file=csv_file,
                    model_idx=model_idx,
                    model_name=model_name,
                    checkpoint_path=checkpoint_path,
                    reference_root=reference_root,
                    times=times,
                    error=e,
                )

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compute recovery and overlap for CIFAR-10 reference reconstructions.")
    parser.add_argument(
        "config",
        type=str,
        help="Pipe-separated config: checkpoint_path | hydra_config_path | reference_dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="recovery_overlap_results.csv",
        help="Output CSV file.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of random reconstruction samples per model and time t.",
    )
    parser.add_argument(
        "--times",
        type=str,
        default="0.0:0.1:1.0",
        help="Mask fractions/times. Use start:step:end or comma-separated values.",
    )
    parser.add_argument(
        "--max-steps-base",
        type=int,
        default=3072,
        help="Base used for sampling steps: steps = round(max_steps_base * t).",
    )
    parser.add_argument(
        "--cfg-gamma",
        type=float,
        default=1.0,
        help="CFG guidance gamma.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Noise schedule epsilon passed to model.reconstruct.",
    )
    parser.add_argument(
        "--hamming-chunk-size",
        type=int,
        default=2048,
        help="Reference chunk size for nearest-Hamming search.",
    )
    parser.add_argument(
        "--l2-chunk-size",
        type=int,
        default=2048,
        help="Reference chunk size for nearest-L2 search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda or cpu. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--reconstruct-t-zero",
        action="store_true",
        help="At t=0, call model.reconstruct with one sampling step instead of copying the original. Recovery is still NaN because N_mask=0.",
    )

    args = parser.parse_args()
    if args.n_samples < 1:
        parser.error("--n-samples must be >= 1")
    if args.max_steps_base < 1:
        parser.error("--max-steps-base must be >= 1")
    if args.hamming_chunk_size < 1:
        parser.error("--hamming-chunk-size must be >= 1")
    if args.l2_chunk_size < 1:
        parser.error("--l2-chunk-size must be >= 1")
    main(args)
