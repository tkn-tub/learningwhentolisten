import numpy as np
from scipy.special import gammaln


def free_diffusion_cir(tau, D, d, volume):
    """Eq. (1): free-diffusion CIR of the communication link."""
    tau = np.asarray(tau, dtype=float)
    out = np.zeros_like(tau)
    valid = tau > 0.0
    tv = tau[valid]
    out[valid] = (
        volume
        / np.power(4.0 * np.pi * D * tv, 1.5)
        * np.exp(-(d ** 2) / (4.0 * D * tv))
    )
    return out


def paper_mean_response(bits, delta_t, cfg):
    """Expected received molecule count for the synchronization frame."""
    bits = np.asarray(bits, dtype=int)
    n = len(bits) * cfg.samples_per_symbol
    t = np.arange(n, dtype=float) * cfg.t_s
    mean = np.full(n, cfg.lambda_bg, dtype=float)

    for k, bit in enumerate(bits):
        if bit:
            mean += cfg.M * free_diffusion_cir(
                t - k * cfg.T - float(delta_t),
                cfg.D,
                cfg.d,
                cfg.V_r,
            )

    return np.maximum(mean, cfg.min_lambda)


def build_timing_templates(bits, cfg):
    return np.asarray(
        [paper_mean_response(bits, dt, cfg) for dt in cfg.delta_grid],
        dtype=float,
    )


def reduced_log_likelihood(y, templates, cfg):
    """Paper Eq. (4), exactly as used in the final manuscript.

    The manuscript uses the synchronization-equivalent reduced objective
    sum_n y_n log(lambda_n).  This is the default paper-reproduction score.
    """
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(templates, dtype=float), cfg.min_lambda)
    return (y[None, :] * np.log(lam)).sum(axis=1)


def full_poisson_log_likelihood(y, templates, cfg):
    """Optional full Poisson score for diagnostics; not the paper default."""
    y = np.asarray(y, dtype=float)
    lam = np.maximum(np.asarray(templates, dtype=float), cfg.min_lambda)
    return (
        y[None, :] * np.log(lam)
        - lam
        - gammaln(y[None, :] + 1.0)
    ).sum(axis=1)


def generate_observation(bits, delta_t, cfg, rng):
    mean = paper_mean_response(bits, delta_t, cfg)
    return rng.poisson(mean).astype(float)
