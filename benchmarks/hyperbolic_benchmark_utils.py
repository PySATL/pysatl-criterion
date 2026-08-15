import numpy as np
import scipy.stats as scipy_stats


ALPHA = 1.5
BETA = 0.25
DELTA = 1.0
MU = 0.0
CALLS = 1_000
SAMPLE_SIZE = 10_000


def create_sorted_sample() -> np.ndarray:
    """Generate one reproducible sorted hyperbolic sample."""
    sample = scipy_stats.genhyperbolic.rvs(
        p=1.0,
        a=ALPHA * DELTA,
        b=BETA * DELTA,
        loc=MU,
        scale=DELTA,
        size=SAMPLE_SIZE,
        random_state=np.random.default_rng(42),
    )
    return np.sort(np.asarray(sample, dtype=np.float64))


def cdf_values(sorted_sample: np.ndarray) -> np.ndarray:
    """Calculate reference CDF values outside the timed statistic kernels."""
    values = scipy_stats.genhyperbolic.cdf(
        sorted_sample,
        p=1.0,
        a=ALPHA * DELTA,
        b=BETA * DELTA,
        loc=MU,
        scale=DELTA,
    )
    return np.asarray(values, dtype=np.float64)


def log_probabilities(sorted_sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate stable tail probabilities outside the timed AD kernels."""
    parameters = {
        "p": 1.0,
        "a": ALPHA * DELTA,
        "b": BETA * DELTA,
        "loc": MU,
        "scale": DELTA,
    }
    log_cdf = scipy_stats.genhyperbolic.logcdf(sorted_sample, **parameters)
    log_sf = scipy_stats.genhyperbolic.logsf(sorted_sample, **parameters)
    return (
        np.asarray(log_cdf, dtype=np.float64),
        np.asarray(log_sf, dtype=np.float64),
    )
