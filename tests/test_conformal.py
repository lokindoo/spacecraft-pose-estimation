import numpy as np
import pytest
from speedplus_pose.conformal import (
    clopper_pearson_interval,
    compute_coverage,
    compute_q_hat,
    coverage_null,
    coverage_null_interval,
    coverage_null_z,
    inflation_factor,
    so3_ball_fraction,
)

# Exact value tests


def test_q_hat_rank():
    ns = [10, 20, 100, 1000, 1000]
    alphas = [0.2, 0.1, 0.1, 0.1, 0.05]
    q_hats = [9, 19, 91, 901, 951]
    for i in range(5):
        assert compute_q_hat(np.arange(1, ns[i] + 1), alphas[i]) == q_hats[i]


def test_q_hat_degeneration():
    assert compute_q_hat(np.arange(1, 11), 0.05) == np.inf


def test_q_hat_monotonicity():
    alphas = [0.5, 0.3, 0.2, 0.1, 0.05]
    previous_q_hat = 0
    scores = np.arange(1, 1000)
    for i in range(5):
        q_hat = compute_q_hat(scores, alphas[i])
        assert q_hat >= previous_q_hat
        previous_q_hat = q_hat


def test_so_3_anchors():
    assert so3_ball_fraction(0) == 0
    assert so3_ball_fraction(np.pi) == 1
    assert np.isclose(so3_ball_fraction(2.31), 0.5, atol=1e-3)
    thetas = [0.05, 0.1, 0.2, 0.3, 0.5]
    previous_so3 = 0
    for i in range(5):
        so3 = so3_ball_fraction(thetas[i])
        assert so3 >= previous_so3
        previous_so3 = so3


def test_compute_coverage_boundaries():
    scores = [1, 2, 3, 4, 5]
    assert compute_coverage(scores, 3) == (3, 5)
    assert compute_coverage(scores, 0) == (0, 5)
    assert compute_coverage(scores, 5) == (5, 5)


def test_clopper_pearson_interval():
    assert clopper_pearson_interval(0, 1)[0] == 0.0
    assert clopper_pearson_interval(10, 10)[1] == 1.0
    count, nobs = 4, 10
    low, high = clopper_pearson_interval(count, nobs)
    assert low <= count / nobs <= high



# Statistical property tests


def _simulate_coverage(draw, n, alpha, trials, seed=0):
    """
    Measure split-conformal coverage by brute force: draw n+1 exchangeable
    samples, calibrate q_hat on the first n, check whether the held-out sample
    falls inside, then repeat and count.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        sample = draw(rng, n + 1)
        hits += sample[n] <= compute_q_hat(sample[:n], alpha)
    return hits / trials


def test_coverage_simulation():
    """
    The guarantee itself, rather than an implementation detail.

    For continuous scores, coverage is exactly k/(n+1) where
    k = ceil((n+1)(1-alpha)). n is deliberately small: at n=19 the correct
    answer is 18/20 = 0.90 and an off-by-one q_hat gives 19/20 = 0.95, a gap of
    ~37 standard errors, whereas at n=1000 the same gap shrinks to about 1 SE
    and becomes invisible. Do not "improve" this test by raising n.
    """
    n, alpha, trials = 19, 0.1, 50_000
    k = int(np.ceil((n + 1) * (1 - alpha)))
    expected = k / (n + 1)
    se = (expected * (1 - expected) / trials) ** 0.5

    coverage = _simulate_coverage(lambda r, m: r.standard_normal(m), n, alpha, trials)

    assert abs(coverage - expected) < 4 * se
    # The published two-sided bound. It is much looser than the point prediction
    # above, and would not catch an off-by-one at this n.
    assert 1 - alpha <= expected <= 1 - alpha + 1 / (n + 1)


def test_distribution_independence():
    """
    Coverage must not depend on the score distribution, which is what makes
    conformal usable on pose errors we have never modelled.

    Cauchy has no mean and no variance; exponential is skewed and positive. Both
    must give the same coverage as the Gaussian case.
    """
    n, alpha, trials = 19, 0.1, 50_000
    k = int(np.ceil((n + 1) * (1 - alpha)))
    expected = k / (n + 1)
    se = (expected * (1 - expected) / trials) ** 0.5

    for draw in [
        lambda r, m: r.standard_cauchy(m),
        lambda r, m: r.exponential(size=m),
    ]:
        coverage = _simulate_coverage(draw, n, alpha, trials)
        assert abs(coverage - expected) < 4 * se

    # Ties break exactness: `<=` also admits scores equal to q_hat, so coverage
    # rises above k/(n+1). The guarantee is one-sided-safe, so it still holds.
    discrete = _simulate_coverage(
        lambda r, m: r.integers(0, 5, size=m).astype(float), n, alpha, trials
    )
    assert discrete >= 1 - alpha


def test_inflation_factor_identity():
    """
    Feeding the same scores as calibration and as "real" must give exactly 1.0.

    This catches an estimator mismatch: using np.percentile for the numerator
    while q_hat uses the corrected order statistic returns 0.988 here rather
    than 1.0. The 2x case pins the direction, so a flipped ratio reads 0.5.
    """
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(1000) ** 2
    q_hat_synth = compute_q_hat(scores, 0.1)

    assert inflation_factor(scores, q_hat_synth, 0.1) == 1.0
    assert np.isclose(inflation_factor(2 * scores, q_hat_synth, 0.1), 2.0)


def _simulate_hit_counts(n_cal, n_test, alpha, trials, seed=0):
    """
    Run the whole split-conformal pipeline `trials` times on a known-good setup
    and return the hit count from each run.

    Calibration and test scores come from the same continuous distribution, so
    exchangeability holds by construction and there is no bug to detect.
    Whatever spread shows up here is irreducible: it is what a correct pipeline
    looks like.
    """
    rng = np.random.default_rng(seed)
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    calibration = rng.standard_normal((trials, n_cal))
    test = rng.standard_normal((trials, n_test))
    q_hat = np.partition(calibration, k - 1, axis=1)[:, k - 1]
    return (test <= q_hat[:, None]).sum(axis=1)


def test_coverage_null_matches_simulation():
    """
    The analytic null must reproduce what the pipeline actually does.
    coverage_null claims the hit count is beta-binomial, so this runs the real
    thing thousands of times and checks that the first two moments agree.
    """
    n_cal, n_test, alpha, trials = 200, 200, 0.1, 4000
    counts = _simulate_hit_counts(n_cal, n_test, alpha, trials)
    distribution = coverage_null(n_cal, n_test, alpha)

    se_of_mean = distribution.std() / np.sqrt(trials)
    assert abs(counts.mean() - distribution.mean()) < 4 * se_of_mean
    assert np.isclose(counts.std(ddof=1), distribution.std(), rtol=0.06)


def test_coverage_null_is_wider_than_binomial():
    """
    Pin the bug this replaced.

    q_hat is estimated rather than given. With equal calibration and test sizes
    the calibration draw contributes about as much variance as the test draw, so
    the honest spread is ~sqrt(2) times the binomial one, and a gate built on the
    binomial spread alone is too narrow by that factor.
    """
    n, alpha = 1000, 0.1
    p = 1 - alpha
    binomial_sd = np.sqrt(p * (1 - p) * n)
    assert np.isclose(coverage_null(n, n, alpha).std() / binomial_sd, np.sqrt(2), rtol=0.02)


def test_gate_false_alarm_rate():
    """
    A gate is only useful if it stays quiet on a correct pipeline.

    Asserted both ways: the beta-binomial region fires at roughly its nominal
    rate, while the Clopper-Pearson interval on the test proportion fires on
    something like one measurement in six. Across 16 measurements, that
    Clopper-Pearson gate is near-certain to fail on a sound pipeline.
    """
    n_cal, n_test, alpha, trials = 200, 200, 0.1, 4000
    counts = _simulate_hit_counts(n_cal, n_test, alpha, trials)

    low, high = coverage_null_interval(n_cal, n_test, alpha, level=0.99)
    beta_binomial_alarms = np.mean((counts < low) | (counts > high))
    assert beta_binomial_alarms <= 0.02

    clopper_pearson_alarms = np.mean([
        not (lambda b: b[0] <= 1 - alpha <= b[1])(clopper_pearson_interval(int(c), n_test))
        for c in counts
    ])
    assert clopper_pearson_alarms > 5 * beta_binomial_alarms


def test_coverage_null_z_and_interval_agree():
    """z near zero at the centre, and widening the level widens the region."""
    n_cal, n_test, alpha = 1000, 1000, 0.1
    centre = int(round(coverage_null(n_cal, n_test, alpha).mean()))
    assert abs(coverage_null_z(centre, n_cal, n_test, alpha)) < 0.5

    narrow = coverage_null_interval(n_cal, n_test, alpha, level=0.95)
    wide = coverage_null_interval(n_cal, n_test, alpha, level=0.999)
    assert wide[0] <= narrow[0] <= centre <= narrow[1] <= wide[1]


def test_coverage_null_degenerate_alpha():
    """alpha below 1/(n+1) makes q_hat infinite; the null is then meaningless."""
    with pytest.raises(AssertionError):
        coverage_null(10, 10, 0.05)


# Guard tests


def test_nan_in_scores():
    """
    A single NaN score must raise rather than propagate.

    Left unguarded, one diverged prediction makes q_hat NaN, every
    `score <= nan` is then False, and coverage reads 0%, which looks like a
    spectacular finding rather than a data problem.
    """
    nan_scores = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(AssertionError):
        compute_q_hat(nan_scores, 0.1)
    with pytest.raises(AssertionError):
        compute_coverage(nan_scores, 1.0)

    inf_scores = np.array([1.0, 2.0, np.inf])
    with pytest.raises(AssertionError):
        compute_q_hat(inf_scores, 0.1)
    with pytest.raises(AssertionError):
        compute_coverage(inf_scores, 1.0)
