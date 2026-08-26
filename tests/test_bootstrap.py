"""Tests for bootstrap CIs, paired significance, Holm, and effect sizes."""

from __future__ import annotations

from depression_rag.evaluation.bootstrap import (
    bootstrap_ci,
    holm_adjust,
    paired_bootstrap_diff,
    rank_biserial,
)


def test_ci_brackets_the_mean_and_is_deterministic():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    a = bootstrap_ci(vals, n_boot=2000, seed=7)
    b = bootstrap_ci(vals, n_boot=2000, seed=7)
    assert a == b                          # seeded -> reproducible
    assert a.low <= a.mean <= a.high
    assert a.mean == sum(vals) / len(vals)
    assert a.n == len(vals)


def test_ci_is_degenerate_for_constant_values():
    ci = bootstrap_ci([1.0] * 20, n_boot=1000)
    assert ci.low == ci.high == ci.mean == 1.0


def test_paired_diff_detects_a_real_gap():
    # a beats b on every query -> diff CI should sit above 0, small p
    a = [1.0] * 30
    b = [0.0] * 30
    t = paired_bootstrap_diff(a, b, n_boot=2000, seed=1)
    assert t.mean_diff == 1.0
    assert t.low > 0
    assert t.p_value <= 0.05


def test_paired_diff_finds_no_gap_when_identical():
    x = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    t = paired_bootstrap_diff(x, x, n_boot=2000, seed=1)
    assert t.mean_diff == 0.0
    assert t.low <= 0 <= t.high          # CI straddles 0 -> not significant
    assert t.p_value == 1.0              # clamped: never exactly 0


def test_paired_requires_alignment():
    import pytest

    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 0.0], [1.0], n_boot=100)


def test_holm_known_example():
    # sorted ps [.005,.01,.03,.04] -> (4).005=.02, (3).01=.03, (2).03=.06,
    # (1).04=.04 -> monotone -> .06; mapped back to input order
    adj = holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert [round(a, 6) for a in adj] == [0.03, 0.06, 0.06, 0.02]


def test_holm_clamps_and_preserves_order():
    adj = holm_adjust([0.5, 0.9])
    assert adj[0] == 1.0 and adj[1] == 1.0   # 2*0.5=1.0, monotone -> 1.0
    assert holm_adjust([]) == []
    single = holm_adjust([0.02])
    assert single == [0.02]                   # m=1: unchanged


def test_rank_biserial_extremes_and_sign():
    assert rank_biserial([0.2, 0.5, 0.1]) == 1.0     # first system wins all pairs
    assert rank_biserial([-0.2, -0.5]) == -1.0       # loses all pairs
    assert rank_biserial([1.0, -1.0]) == 0.0         # symmetric (tied |d|)
    assert rank_biserial([0.0, 0.0]) == 0.0          # zero diffs dropped
    assert rank_biserial([0.0, 0.3]) == 1.0
