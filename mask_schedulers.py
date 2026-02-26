## Mask Schedulers for iterative unmasking in masked language modeling.
##
## Public API: UnmaskOptimalRandomScheduler,
##              UnmaskUnifKScheduler,
##              UnmaskLowEntropyKScheduler,
##              UnmaskEBScheduler

import torch
import pandas as pd
import numpy as np
from pathlib import Path


def get_scheduler(config, mask_index):
    """Instantiate the mask scheduler specified in config.sampling.sampler.

    Supported values:
      - 'mdlm-k'       -> UnmaskUnifKScheduler(k = seq_len // steps)
      - 'size-optimal' -> UnmaskOptimalRandomScheduler(num_steps = steps)
      - 'top-k-entropy'  -> UnmaskLowEntropyKScheduler(k = seq_len // steps)
      - 'eb-entropy'   -> UnmaskEBScheduler(gamma = sampling.gamma_eb)
    """
    sampler = config.sampling.sampler
    num_steps = config.sampling.steps
    seq_length = config.model.length
    ## TODO nucleus_p not implemented in schedulers yet, but we can use it to load the appropriate info profile for the size-optimal scheduler.  For now we just assert that it's 0.9 since that's the only profile we have generated so far.
    # topp = config.sampling.nucleus_p # only used to load the appropriate info profile for the size-optimal scheduler,
    #                                  # but doesn't affect the actual schedulers returned by this function.
    #                                  # Nucleus p thresholding is taken care of outside the schedulers logic.
    if sampler == 'mdlm-k':
        assert seq_length % num_steps == 0, \
            'Sequence length must be divisible by num_steps for mdlm-k.'
        return UnmaskUnifKScheduler(mask_index=mask_index, k=seq_length // num_steps)
    elif sampler == 'size-optimal':
        return UnmaskOptimalRandomScheduler(mask_index=mask_index, num_steps=num_steps, topp=topp)
    elif sampler == 'top-k-entropy':
        assert seq_length % num_steps == 0, \
            'Sequence length must be divisible by num_steps for top-k-entropy.'
        return UnmaskLowEntropyKScheduler(mask_index=mask_index, k=seq_length // num_steps)
    elif sampler == 'eb-entropy':
        return UnmaskEBScheduler(mask_index=mask_index, gamma=config.sampling.gamma_eb)
    else:
        raise ValueError(f'Unknown sampler: {sampler}. '
                         'Choose from mdlm-k, size-optimal, top-k-entropy, eb-entropy.')


class AbstractMaskScheduler:
    def __init__(self):
        pass
    
    def compute_pos_to_unmask(self, px0, xt):
        raise NotImplementedError


PRJ_ROOT = Path(__file__).resolve().parents[0]

class UnmaskOptimalRandomScheduler(AbstractMaskScheduler):
    """
    At each step, unmask k positions at random.
    """
    def __init__(self, mask_index, num_steps, topp = 0.9, profile_path = None):
        super().__init__()
        self.mask_index = mask_index
        self.num_steps = num_steps
        assert isinstance(topp, float) and 0.0 < topp <= 1.0, "topp must be a float in (0, 1]"
        assert topp in (0.9, 1.0), "Only topp=0.9 and topp=1.0 are supported for the size-optimal scheduler since those are the only info profiles we have generated so far."
        if profile_path is None:
            profile_path = PRJ_ROOT / f"results_analysis/mdlm-k_topp{topp}_info_profile.csv"
        df_profile = pd.read_csv(profile_path)
        self.info_profile = df_profile["info_profile_interpolated"].values
        self.L = len(self.info_profile)
        assert self.L == 1024, f"Info profile length {self.L} does not match expected sequence length 1024."
        self.a, self.s = partition_sizes_from_infoprofile(self.info_profile, self.num_steps)

    def get_pos_to_unmask(self, p_x0, xt):
        batch_size, seq_len = xt.shape
        masked = (xt == self.mask_index)  # [batch_size, seq_len]
        num_masked = masked.sum(dim=1).cpu().numpy()  # [batch_size]
        # assert all same num masked
        nm = num_masked[0]
        assert all(nm == nmi for nmi in num_masked), "All sequences must have same number of masked positions"
        breakpoint = self.L - nm
        # assign step to the index k where a[k] == breakpoint
        curr_step = None
        for ki in range(len(self.a)-1):
            if self.a[ki] == breakpoint:
                curr_step = ki
                break
        assert curr_step is not None, "Could not find appropriate step for breakpoint"
        k = self.s[curr_step]  # number of positions to unmask this step
        pos_to_unmask = []      # list of posidx of length batch_size
        for b in range(batch_size):
            idx = torch.nonzero(masked[b], as_tuple=False).squeeze(1)  # masked positions in this row
            nm = idx.numel()
            if nm == 0:
                pos_to_unmask.append([])
                continue

            nk = min(k, nm)
            # Choose nk distinct masked positions uniformly
            perm = torch.randperm(nm, device=xt.device)[:nk]
            pos_idx = idx[perm].tolist()                    # [nk]
            pos_to_unmask.append(pos_idx)

        return pos_to_unmask

def partition_sizes_from_infoprofile(f, K):
    """
    Simple O(K*N^2) dynamic programming solution for:
      maximize S = sum_{k=0}^{K-1} (a_{k+1}-a_k)*f(a_k)
    subject to:
      a0 = 0, aK = N, integers, strictly increasing.

    Parameters
    ----------
    f : array-like, length N
        f(i) for i=0..N-1
    K : int
        number of segments (so there are K+1 breakpoints)

    Returns
    -------
    a: list of breakpoints
        a[0] = 0, a[K] = N, strictly increasing
    s : list of patition sizes
        s[k] = a_{k+1} - a_k for k=0..K-1
    """
    f = np.asarray(f, dtype=float)
    N = len(f)

    # dp_prev[j] = dp[t-1, j]
    dp_prev = np.full(N + 1, -np.inf)
    dp_prev[0] = 0.0

    # to reconstruct solution: parent[t, j] = argmax i for dp[t, j]
    parent = np.full((K + 1, N + 1), -1, dtype=int)

    for t in range(1, K + 1):
        dp_cur = np.full(N + 1, -np.inf)

        # endpoints must be >= t because strictly increasing from 0:
        # minimal is [0,1,2,...,t]
        for j in range(t, N + 1):
            best_val = -np.inf
            best_i = -1

            # i must be at least t-1 for feasibility, and < j
            for i in range(t - 1, j):
                if i == N:  # not really possible since i<j<=N
                    continue
                # f(i) is defined only for i in [0..N-1]
                if i <= N - 1:
                    val = dp_prev[i] + (j - i) * f[i]
                    if val > best_val:
                        best_val = val
                        best_i = i

            dp_cur[j] = best_val
            parent[t, j] = best_i

        dp_prev = dp_cur

    S_star = dp_prev[N]
    
    # backtrack breakpoints
    a = np.empty(K + 1, dtype=int)
    a[K] = N
    j = N
    for t in range(K, 0, -1):
        i = parent[t, j]
        a[t - 1] = i
        j = i
    a[0] = 0
    s = [a[i+1]-a[i] for i in range(len(a)-1)]
    return a, s

class UnmaskUnifKScheduler(AbstractMaskScheduler):
    """
    At each step, unmask k positions at random.
    """
    def __init__(self, mask_index, k):
        super().__init__()
        self.mask_index = mask_index
        self.k = k

    def get_pos_to_unmask(self, p_x0, xt):
        batch_size, seq_len = xt.shape
        masked = (xt == self.mask_index)  # [batch_size, seq_len]
    
        pos_to_unmask = []      # list of posidx of length batch_size
    
        for b in range(batch_size):
            idx = torch.nonzero(masked[b], as_tuple=False).squeeze(1)  # masked positions in this row
            nmasked = idx.numel()
            if nmasked == 0:
                pos_to_unmask.append([])
                continue

            nk = min(self.k, nmasked)
            # Choose nk distinct masked positions uniformly
            perm = torch.randperm(nmasked, device=xt.device)[:nk]
            pos_idx = idx[perm].tolist()                    # [nk]
            pos_to_unmask.append(pos_idx)

        return pos_to_unmask


class UnmaskLowEntropyKScheduler(AbstractMaskScheduler):
    """
    At each step, unmask the K masked positions with the lowest entropy
    in the model's predicted distribution p_x0.
    """
    def __init__(self, mask_index, k):
        super().__init__()
        self.mask_index = mask_index
        self.k = k

    def get_pos_to_unmask(self, p_x0, xt):
        batch_size, _ = xt.shape
        masked = (xt == self.mask_index)  # [batch_size, seq_len]

        pos_to_unmask = []

        for b in range(batch_size):
            idx = torch.nonzero(masked[b], as_tuple=False).squeeze(1)  # masked positions
            nmasked = idx.numel()
            if nmasked == 0:
                pos_to_unmask.append([])
                continue

            # Entropy at each masked position: -sum(p * log(p))
            p = p_x0[b, idx]  # [nmasked, vocab_size]
            entropy = -(p.clamp(min=1e-10) * p.clamp(min=1e-10).log()).sum(dim=-1)  # [nmasked]

            nk = min(self.k, nmasked)
            _, top_idx = torch.topk(entropy, nk, largest=False)
            pos_to_unmask.append(idx[top_idx].tolist())

        return pos_to_unmask


class UnmaskEBScheduler(AbstractMaskScheduler):
    """
    Entropy Bounded (EB) Sampler from Ben-Hamu et al. (2025),
    "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking".

    At each step, sorts masked positions by entropy (ascending) and unmasks
    the largest prefix U of that sorted order satisfying:

        sum(H[U]) - max(H[U]) <= gamma

    Since the positions are sorted ascending, max(H[U]) = H[last], so the
    criterion checks that the sum of all entropies except the largest stays
    within gamma.  This upper-bounds the joint dependence error introduced
    by sampling the positions in U independently.

    gamma=0  -> unmask exactly 1 token per step (pure greedy entropy order).
    gamma=inf -> unmask all masked tokens at once.
    """
    def __init__(self, mask_index, gamma):
        super().__init__()
        self.mask_index = mask_index
        self.gamma = gamma

    def get_pos_to_unmask(self, p_x0, xt):
        batch_size, _ = xt.shape
        masked = (xt == self.mask_index)  # [batch_size, seq_len]

        pos_to_unmask = []

        for b in range(batch_size):
            idx = torch.nonzero(masked[b], as_tuple=False).squeeze(1)  # masked positions
            nmasked = idx.numel()
            if nmasked == 0:
                pos_to_unmask.append([])
                continue

            # Entropy at each masked position: -sum(p * log(p))
            p = p_x0[b, idx]  # [nmasked, vocab_size]
            entropy = -(p.clamp(min=1e-10) * p.clamp(min=1e-10).log()).sum(dim=-1)  # [nmasked]

            # Sort ascending: low entropy (confident) positions come first
            sorted_entropy, sort_order = torch.sort(entropy)

            # EB criterion: for sorted positions, after adding position i the
            # cumulative bound is sum(H[0..i]) - H[i] = sum(H[0..i-1]) <= gamma.
            # Since the sequence is sorted ascending, cummax[i] == sorted_entropy[i].
            acc_entropy = torch.cumsum(sorted_entropy, dim=0)
            cummax_entropy = torch.cummax(sorted_entropy, dim=0).values
            k = int((acc_entropy - cummax_entropy <= self.gamma).sum().item())
            k = max(1, min(k, nmasked))  # always unmask at least 1

            selected = sort_order[:k]
            pos_to_unmask.append(idx[selected].tolist())

        return pos_to_unmask


