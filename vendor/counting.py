from typing import Tuple
import numpy as np

def bin_index(s: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map s to bin index in [0, nbins-1], out-of-range -> -1."""
    nbins = len(edges) - 1
    idx = np.searchsorted(edges, s, side="right") - 1
    idx = idx.astype(int, copy=False)
    bad = (idx < 0) | (idx >= nbins) | ~np.isfinite(s)
    idx[bad] = -1
    return idx

def wrap_periodic(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Wrap x into [lo, hi) periodically."""
    L = hi - lo
    if not (L > 0):
        raise ValueError("invalid periodic interval")
    y = (x - lo) % L + lo
    # Ensure hi maps to lo
    y = np.where(y >= hi, y - L, y)
    return y

def transition_counts_hummer(
    s: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    lag: int,
    stride: int,
    center_shift: bool,
    periodic: bool,
) -> Tuple[np.ndarray, int, int]:
    """
    Build N_ij counts where N_ij is transitions from j -> i after lag frames.
    Returns (N, n_valid, n_far), with n_far = count(|Δbin|>1) for diagnostic.
    Implements Hummer's bin-center shift (below Eq. 15) if center_shift=True:
        j = bin(s0)
        s0c = center[j]
        s1' = s0c + (s1 - s0)
        i = bin(s1')
    """
    nbins = len(edges) - 1
    N = np.zeros((nbins, nbins), dtype=np.int64)

    if lag < 1 or stride < 1 or s.size <= lag:
        return N, 0, 0

    idx0 = np.arange(0, s.size - lag, stride, dtype=int)
    s0 = s[idx0]
    s1 = s[idx0 + lag]

    j = bin_index(s0, edges)
    good = (j >= 0)
    if not np.any(good):
        return N, 0, 0

    s0 = s0[good]
    s1 = s1[good]
    j = j[good]

    if center_shift:
        s0c = centers[j]
        s1p = s0c + (s1 - s0)
    else:
        s1p = s1

    if periodic:
        s1p = wrap_periodic(s1p, edges[0], edges[-1])

    i = bin_index(s1p, edges)

    good2 = (i >= 0)
    i = i[good2]
    j = j[good2]

    n_valid = int(i.size)
    if n_valid == 0:
        return N, 0, 0

    # diagnostic: far jumps in bin space
    n_far = int(np.sum(np.abs(i - j) > 1))

    # accumulate
    np.add.at(N, (i, j), 1)
    return N, n_valid, n_far
