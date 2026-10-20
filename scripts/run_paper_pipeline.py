"""Paper-aligned SmartSynch reproduction pipeline.

This script is intentionally compact and mirrors the final manuscript flow.
It is not executed automatically by the repository.
"""

from pathlib import Path
import numpy as np

from smartsynch.config import PaperConfig
from smartsynch.data import generate_positive_dataset
from smartsynch.models import DualRCSystem
from smartsynch.evaluation import likelihood_fidelity, sto_table
from smartsynch.plotting import plot_surrogate_fidelity, plot_sto_ecdf


def main():
    cfg = PaperConfig()
    cfg.validate()

    rng = np.random.default_rng(cfg.seed)

    train = generate_positive_dataset(cfg.n_train, cfg.synch_frame, cfg, rng)
    test = generate_positive_dataset(cfg.n_test, cfg.synch_frame, cfg, rng)

    system = DualRCSystem(cfg).fit(train)

    print("Paper configuration")
    print("  T =", cfg.T, "s")
    print("  t_s =", cfg.t_s, "s")
    print("  N =", cfg.N)
    print("  J =", cfg.J)
    print("  D =", cfg.D, "um^2/s")
    print("  d =", cfg.d, "um")
    print("  d_RC =", cfg.d_rc, "um")
    print("  rho(W) =", system.rc_phy.reservoir.spectral_radius)
    print("  beta =", cfg.ridge_beta)

    fidelity, exact_all, rc_all = likelihood_fidelity(test, system.rc_phy)
    sto, truth, estimates = sto_table(test, system)

    out = Path("results")
    out.mkdir(exist_ok=True)

    fidelity.to_csv(out / "rc_phy_fidelity.csv", index=False)
    sto.to_csv(out / "sto_summary.csv", index=False)

    plot_surrogate_fidelity(
        exact_all,
        rc_all,
        out=out / "surrogate_fidelity.pdf",
        seed=cfg.seed,
    )
    plot_sto_ecdf(
        truth,
        estimates,
        out=out / "sto_ecdf.pdf",
    )

    print("\nRC-PHY fidelity")
    print(fidelity.describe())
    print("\nSTO summary")
    print(sto)


if __name__ == "__main__":
    main()
