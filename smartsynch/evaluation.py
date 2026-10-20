import numpy as np
import pandas as pd

from .physics import reduced_log_likelihood


def exact_estimate(y, templates, cfg):
    score = reduced_log_likelihood(y, templates, cfg)
    return float(cfg.delta_grid[int(np.argmax(score))])


def threshold_estimate(y, cfg):
    """Lightweight threshold baseline used only as a reproducible proxy.

    The final paper describes a normalized peak/detection threshold construction
    but does not provide every implementation detail needed to reconstruct the
    exact original code.  This implementation is deliberately simple and is
    labeled as such in the project documentation.
    """
    y = np.asarray(y, dtype=float)
    yn = y / (np.max(y) + 1e-12)
    threshold = 0.5
    hits = np.flatnonzero(yn >= threshold)
    if len(hits) == 0:
        return 0.0
    t = float((hits[0] % cfg.samples_per_symbol) * cfg.t_s)
    j = int(np.argmin(np.abs(cfg.delta_grid - t)))
    return float(cfg.delta_grid[j])


def likelihood_fidelity(samples, rc_phy):
    rows = []
    exact_all, rc_all = [], []

    for s in samples:
        ex = reduced_log_likelihood(s["y"], rc_phy.templates, rc_phy.cfg)
        pr = rc_phy.score_landscape(s["y"])
        exact_all.append(ex)
        rc_all.append(pr)

        rmse = float(np.sqrt(np.mean((ex - pr) ** 2)))
        denom = float(np.max(ex) - np.min(ex) + 1e-12)
        rows.append(
            {
                "RMSE": rmse,
                "NRMSE": rmse / denom,
                "ArgmaxAgreement": float(np.argmax(ex) == np.argmax(pr)),
            }
        )

    return (
        pd.DataFrame(rows),
        np.asarray(exact_all),
        np.asarray(rc_all),
    )


def sto_table(samples, system):
    cfg = system.cfg
    truth = np.asarray([s["delta_true"] for s in samples], dtype=float)

    methods = {
        "Exact-Physics-ML": lambda y: exact_estimate(y, system.rc_phy.templates, cfg),
        "Direct-RC (Argmax)": lambda y: system.rc_phy.estimate_argmax(y),
        "Proposed RC-RL": lambda y: system.estimate(y),
        "Thresholding": lambda y: threshold_estimate(y, cfg),
    }

    estimates = {}
    rows = []

    for name, fn in methods.items():
        est = np.asarray([fn(s["y"]) for s in samples], dtype=float)
        estimates[name] = est
        ae = np.abs(est - truth)
        rows.append(
            {
                "Method": name,
                "MAE_s": float(np.mean(ae)),
                "MedianAE_s": float(np.median(ae)),
                "P90_AE_s": float(np.quantile(ae, 0.90)),
            }
        )

    return pd.DataFrame(rows), truth, estimates
