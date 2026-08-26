"""Bootstrap confidence intervals + paired significance for the results table.

Retrieval metrics here are per-query means over a fixed question set, so:

* a **percentile bootstrap** over queries gives each index's metric a CI without
  distributional assumptions (the per-query values are 0/1 hits or bounded nDCG,
  not normal);
* a **paired bootstrap** over the SAME queries tests whether one index really beats
  another — pairing removes per-question difficulty as a nuisance, which an
  unpaired test would leave in.

Deterministic given a seed (project reproducibility value): resampling uses a
seeded numpy Generator, so reported CIs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_N_BOOT = 10000
DEFAULT_SEED = 20260628  # matches the pipeline run seed


@dataclass(frozen=True)
class CI:
    mean: float
    low: float
    high: float
    n: int


def bootstrap_ci(
    values: list[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> CI:
    """Percentile bootstrap CI for the mean of per-query values."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(x.mean()), float(lo), float(hi), int(n))


@dataclass(frozen=True)
class PairedTest:
    mean_diff: float          # mean(a) - mean(b)
    low: float
    high: float
    p_value: float            # two-sided bootstrap p for diff != 0
    n: int


def paired_bootstrap_diff(
    a: list[float],
    b: list[float],
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> PairedTest:
    """Paired bootstrap on the per-query difference a[i] - b[i] (same queries).

    ``a`` and ``b`` must be aligned by query. Returns the mean difference, its CI,
    and a two-sided bootstrap p-value (twice the smaller tail mass at 0, clamped to
    [1/n_boot, 1] so it is never reported as exactly 0).
    """
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    if da.shape != db.shape:
        raise ValueError(f"paired arrays must align: {da.shape} vs {db.shape}")
    d = da - db
    n = d.size
    if n == 0:
        return PairedTest(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    # two-sided p: 2 * mass on the side of 0 opposite the observed mean diff
    tail = np.mean(boot <= 0) if d.mean() > 0 else np.mean(boot >= 0)
    p = min(1.0, 2.0 * max(tail, 1.0 / n_boot))
    return PairedTest(float(d.mean()), float(lo), float(hi), float(p), int(n))


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment for multiple comparisons.

    Returns the adjusted p-values in the ORIGINAL order: sort ascending, multiply
    the i-th smallest by (m - i), enforce monotonicity, clamp to 1. Rejecting
    adjusted p < alpha controls the family-wise error rate at alpha with no
    independence assumption.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return []
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def rank_biserial(diffs: list[float]) -> float:
    """Signed rank-biserial correlation — the effect size reported alongside a
    Wilcoxon signed-rank test.

    r = (W+ − W−) / (W+ + W−) over the ranks of |diff| (zero diffs dropped,
    ties given their average rank — matching ``scipy.stats.wilcoxon`` with
    ``zero_method="wilcox"``). +1 = first system wins every pair, −1 = loses
    every pair, 0 = symmetric.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0.0]
    if d.size == 0:
        return 0.0
    ranks = _average_ranks(np.abs(d))
    pos = float(ranks[d > 0].sum())
    neg = float(ranks[d < 0].sum())
    return (pos - neg) / (pos + neg)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (midranks)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=float)
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]
