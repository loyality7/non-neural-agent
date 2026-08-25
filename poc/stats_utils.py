"""Statistical significance utilities for comparing agent results across
seeds.

Implements a stdlib-only bootstrap confidence interval and Cohen's d, so
that comparisons between agents are backed by formal significance testing
rather than a raw comparison of mean values, which can be misleading given
sample-to-sample variance.
"""

import random
import statistics


def bootstrap_diff_ci(sample_a, sample_b, n_resamples=5000, ci=0.95, seed=0):
    """95% CI on mean(sample_a) - mean(sample_b) via bootstrap resampling.
    If the CI excludes zero, the difference is not explainable by sampling
    noise alone at this seed count. Returns (diff_mean, lo, hi).
    """
    rng = random.Random(seed)
    diffs = []
    n_a, n_b = len(sample_a), len(sample_b)
    for _ in range(n_resamples):
        resample_a = [sample_a[rng.randrange(n_a)] for _ in range(n_a)]
        resample_b = [sample_b[rng.randrange(n_b)] for _ in range(n_b)]
        diffs.append(statistics.mean(resample_a) - statistics.mean(resample_b))
    diffs.sort()
    alpha = 1 - ci
    lo_idx = int(n_resamples * (alpha / 2))
    hi_idx = int(n_resamples * (1 - alpha / 2)) - 1
    diff_mean = statistics.mean(sample_a) - statistics.mean(sample_b)
    return round(diff_mean, 3), round(diffs[lo_idx], 3), round(diffs[hi_idx], 3)


def significant_improvement(sample_a, sample_b, **kwargs):
    """True only if the bootstrap CI on (a - b) excludes zero AND is
    entirely positive -- i.e. a is reliably greater than b, not just on
    average across this particular batch of seeds.
    """
    diff_mean, lo, hi = bootstrap_diff_ci(sample_a, sample_b, **kwargs)
    return lo > 0, (diff_mean, lo, hi)


def cohens_d(sample_a, sample_b):
    """Standardized effect size. <0.2 negligible, ~0.5 medium, >0.8 large --
    report this alongside significance, since a significant-but-tiny effect
    with 20 seeds is still a weak result and should be described as such.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    mean_a, mean_b = statistics.mean(sample_a), statistics.mean(sample_b)
    var_a = statistics.variance(sample_a) if n_a > 1 else 0.0
    var_b = statistics.variance(sample_b) if n_b > 1 else 0.0
    pooled_sd = (((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)) ** 0.5
    if pooled_sd == 0:
        return float("inf") if mean_a != mean_b else 0.0
    return round((mean_a - mean_b) / pooled_sd, 3)
