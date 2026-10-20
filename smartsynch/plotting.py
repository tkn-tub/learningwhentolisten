import numpy as np
import matplotlib.pyplot as plt


def plot_surrogate_fidelity(exact_all, rc_all, out=None, seed=42):
    x = np.asarray(exact_all).ravel()
    y = np.asarray(rc_all).ravel()

    rng = np.random.default_rng(seed)
    if len(x) > 5000:
        ids = rng.choice(len(x), 5000, replace=False)
        x, y = x[ids], y[ids]

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())

    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=300)
    ax.scatter(x, y, s=12, alpha=0.30, edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.4, label="Ideal fidelity")
    ax.set_xlabel(r"Exact Physical Likelihood $\Lambda_{\mathrm{PHY}}$")
    ax.set_ylabel(r"Direct RC Surrogate $\hat{\Lambda}^{\mathrm{RC}}$")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def plot_sto_ecdf(truth, estimates, out=None):
    fig, ax = plt.subplots(figsize=(5.8, 4.6), dpi=300)

    for name in [
        "Exact-Physics-ML",
        "Direct-RC (Argmax)",
        "Proposed RC-RL",
        "Thresholding",
    ]:
        ae = np.sort(np.abs(np.asarray(estimates[name]) - truth))
        ecdf = np.arange(1, len(ae) + 1) / len(ae)
        ax.plot(np.r_[0.0, ae], np.r_[0.0, ecdf], label=name, linewidth=1.8)

    ax.set_xlabel(r"Absolute STO Error $|\hat{\Delta t}-\Delta t|$ [s]")
    ax.set_ylabel("Empirical CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig
