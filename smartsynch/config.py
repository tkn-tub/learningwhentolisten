from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class PaperConfig:
    """Parameters reported in the final manuscript."""

    seed: int = 42

    # Communication link / timing grid
    T: float = 4.0
    t_s: float = 0.1
    M: float = 8000.0
    D: float = 3.3
    receiver_radius: float = 4.0
    d: float = 11.0
    delta_step: float = 0.1

    # Physical reservoir
    N: int = 40
    d_rc: float = 3.0
    k_on_molar: float = 6.022e8
    k_off: float = 1.0
    spectral_radius: float = 0.89

    # Readout
    ridge_beta: float = 3e-4

    # Numerical safeguards / optional background
    lambda_bg: float = 1e-9
    min_lambda: float = 1e-12
    reservoir_memory_symbols: int = 1
    normalize_reservoir_drive: bool = True

    # Dataset sizes are engineering defaults, not paper claims.
    n_train: int = 1200
    n_val: int = 300
    n_test: int = 400

    # A short synchronization sequence for reproducible examples.
    # The paper does not numerically specify the exact training preamble.
    synch_frame: np.ndarray = field(
        default_factory=lambda: np.array([1, 1, 0, 0, 1], dtype=int)
    )

    @property
    def samples_per_symbol(self) -> int:
        return int(round(self.T / self.t_s))

    @property
    def delta_grid(self) -> np.ndarray:
        return np.arange(0.0, self.T, self.delta_step)

    @property
    def J(self) -> int:
        return len(self.delta_grid)

    @property
    def V_r(self) -> float:
        return 4.0 / 3.0 * np.pi * self.receiver_radius ** 3

    def validate(self) -> None:
        assert self.samples_per_symbol == 40, "Paper configuration requires N=40 samples/symbol."
        assert self.N == 40, "Paper configuration uses N=40 reservoir nodes."
        assert self.J == 40, "Paper configuration uses J=40 timing candidates."
        assert 0.0 < self.spectral_radius < 1.0
