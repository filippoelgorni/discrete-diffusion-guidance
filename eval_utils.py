import os
import typing

import numpy as np
import torch
import transformers
from scipy.spatial.distance import cdist
from tqdm import tqdm

import diffusion


def compute_ppl(
    pretrained_model,
    val_ds
):
  ppl_metrics = diffusion.Perplexity().to('cuda')
  pbar = tqdm(val_ds, desc='PPL')
  for batch in pbar:
    input_ids = batch['input_ids'].to('cuda')
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask'].to('cuda')
    else:
      attention_mask = None
    losses = pretrained_model._loss(input_ids, attention_mask)
    ppl_metrics.update(losses.nlls, losses.token_mask)
    pbar.set_postfix({'ppl': ppl_metrics.compute().item()})
  return ppl_metrics.compute().item()


def compute_generative_ppl(
    sentences,
    eval_model_name_or_path,
    gen_ppl_eval_batch_size=8,
    max_length=128):
  gen_ppl_metric = diffusion.Perplexity().to('cuda')
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
    eval_model_name_or_path)
  if eval_model_tokenizer.pad_token is None:
    eval_model_tokenizer.pad_token = \
      eval_model_tokenizer.eos_token
    eval_model_tokenizer.pad_token_id = \
      eval_model_tokenizer.eos_token_id
  eval_model = transformers.AutoModelForCausalLM.from_pretrained(
    eval_model_name_or_path).eval()
  if max_length is None:
    max_length = max_length
  eval_model = eval_model.to('cuda')
  # Re-tokenize using eval model's tokenizer
  tokenizer_kwargs = {
    'return_tensors': 'pt',
    'return_token_type_ids': False,
    'return_attention_mask': True,
    'truncation': True,
    'padding': True,
    'max_length': max_length,
  }
  eval_context_size = 1024
  samples = eval_model_tokenizer(
    sentences, **tokenizer_kwargs)
  attn_mask = samples['attention_mask']
  samples = samples['input_ids']
  attn_mask = attn_mask.to('cuda')
  samples = samples.to('cuda')
  num_batches = samples.shape[0] // gen_ppl_eval_batch_size
  for i in tqdm(range(num_batches),
                desc='Gen. PPL', leave=False):
    _samples = torch.split(
      samples[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    _attn_mask = torch.split(
      attn_mask[i * gen_ppl_eval_batch_size: (i + 1) * gen_ppl_eval_batch_size],
      eval_context_size,
      dim=-1)
    for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
      logits = eval_model(
        sample_chunk, attention_mask=attn_mask_chunk)[0]
      logits = logits.transpose(-1, -2)

      nlls = torch.nn.functional.cross_entropy(
        logits[..., :-1],
        sample_chunk[..., 1:],
        reduction='none')
      # first_eos = (sample_chunk == eval_model_tokenizer.eos_token_id).cumsum(-1) == 1
      # token_mask = (sample_chunk != eval_model_tokenizer.eos_token_id)
      # gen_ppl_metric.update(
      #   nlls, first_eos[..., 1:] + token_mask[..., 1:])
      gen_ppl_metric.update(
        nlls, attn_mask_chunk[..., 1:])
  return gen_ppl_metric.compute().item()


def compute_memorization_fast(
    generated: torch.Tensor,
    reference: torch.Tensor,
    k: float = 1/3,
    chunk_size: int = 500,
    generated_classes: typing.Optional[np.ndarray] = None,
    reference_classes: typing.Optional[np.ndarray] = None,
) -> typing.Tuple[float, np.ndarray, np.ndarray]:
  """
  Compute memorization metric f_mem efficiently.
  
  A sample is memorized if:
      ||x - a^μ1||_2 / ||x - a^μ2||_2 < k
  
  Args:
      generated: Generated samples (N, C, H, W) or (N, L) flattened
      reference: Reference images (M, C, H, W) or (M, L) flattened
      k: Threshold for memorization (default 1/3)
      chunk_size: Process in chunks to avoid OOM
      generated_classes: Optional class labels for generated samples (N,)
      reference_classes: Optional class labels for reference samples (M,)
  
  Returns:
      f_mem: Fraction of memorized samples
      nearest_indices: Index of nearest neighbor for each sample
      memorization_ratios: Ratio for each sample
  """
  # Flatten if images
  if len(generated.shape) == 4:
    gen_flat = generated.view(generated.shape[0], -1).cpu().numpy()
  else:
    gen_flat = generated.cpu().numpy()
  
  if len(reference.shape) == 4:
    ref_flat = reference.view(reference.shape[0], -1).cpu().numpy()
  else:
    ref_flat = reference.cpu().numpy()
  
  nearest_indices = []
  memorization_ratios = []
  
  # If class labels provided, compute per-class for efficiency
  use_class_filtering = (
    generated_classes is not None and 
    reference_classes is not None
  )
  
  for i in range(0, len(gen_flat), chunk_size):
    chunk = gen_flat[i:i + chunk_size]
    chunk_indices = np.arange(i, min(i + chunk_size, len(gen_flat)))
    
    # Determine reference samples to compare against
    if use_class_filtering:
      chunk_classes = generated_classes[chunk_indices]
      # For each generated sample, use reference from same class only
      chunk_distances = []
      chunk_nearest = []
      chunk_ratios = []
      
      for j, sample_class in enumerate(chunk_classes):
        # Get indices of reference images from same class
        same_class_mask = reference_classes == sample_class
        if not same_class_mask.any():
          # Fallback: use all reference if no same-class available
          same_class_ref = ref_flat
        else:
          same_class_ref = ref_flat[same_class_mask]
        
        # Compute distances for this sample
        distances = cdist(chunk[j:j+1], same_class_ref, metric='euclidean')[0]
        
        if len(distances) >= 2:
          sorted_indices = np.argsort(distances)
          d1 = distances[sorted_indices[0]]
          d2 = distances[sorted_indices[1]]
        else:
          # Not enough reference samples, mark as not memorized
          d1 = 1.0
          d2 = 2.0
        
        ratio = d1 / (d2 + 1e-8)
        chunk_nearest.append(0)  # Relative index not meaningful with class filtering
        chunk_ratios.append(ratio)
      
      nearest_indices.extend(chunk_nearest)
      memorization_ratios.extend(chunk_ratios)
    else:
      # Original implementation: compare against all reference
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
      memorization_ratios.extend(ratio.tolist())
  
  nearest_indices = np.array(nearest_indices)
  memorization_ratios = np.array(memorization_ratios)
  
  # Count memorized samples
  memorized = memorization_ratios < k
  f_mem = memorized.sum() / len(memorized)
  
  return float(f_mem), nearest_indices, memorization_ratios
