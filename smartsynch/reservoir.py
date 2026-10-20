from collections import deque
import numpy as np

from .physics import free_diffusion_cir


class PhysicalDelayLineReservoir:
    """Linearized molecular physical reservoir from Eqs. (7)–(11).

    W has the paper's delay-line structure and spectral radius rho(W)=|alpha|.
    The molecular reservoir-link CIR supplies the fixed physical input mask.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.N = cfg.N
        self.alpha = float(cfg.spectral_radius)

        self.W = np.zeros((self.N, self.N), dtype=float)
        self.W[0, 0] = self.alpha
        if self.N > 1:
            self.W[1:, :-1] = np.eye(self.N - 1)

        self._build_reservoir_cir()

    @property
    def spectral_radius(self):
        return float(np.max(np.abs(np.linalg.eigvals(self.W))))

    def _build_reservoir_cir(self):
        L = self.cfg.samples_per_symbol * self.cfg.reservoir_memory_symbols
        # h(0)=0 under the causal free-diffusion model.
        tau = np.arange(L, dtype=float) * self.cfg.t_s
        self.h_rc = free_diffusion_cir(
            tau,
            D=self.cfg.D,
            d=self.cfg.d_rc,
            volume=1.0,
        )
        self.L = L

    def _physical_drive(self, u):
        """CIR-filtered concentration-like input drive."""
        u = np.asarray(u, dtype=float)
        history = deque([0.0] * self.L, maxlen=self.L)
        drive = np.zeros_like(u, dtype=float)

        for n, value in enumerate(u):
            history.appendleft(float(value))
            h = np.asarray(history, dtype=float)
            drive[n] = float(np.dot(self.h_rc, h))

        if self.cfg.normalize_reservoir_drive:
            scale = np.max(np.abs(drive))
            if scale > 0:
                drive = drive / scale

        return drive

    def transform(self, u):
        """Return full state trajectory X[n] in R^N."""
        drive = self._physical_drive(u)
        X = np.zeros((len(drive), self.N), dtype=float)
        x = np.zeros(self.N, dtype=float)

        for n, c_n in enumerate(drive):
            x_new = np.empty_like(x)
            # Linearized first receptor state; alpha captures Eq. (18)/(11).
            x_new[0] = self.alpha * x[0] + (1.0 - self.alpha) * c_n
            # Remaining virtual nodes are the occupancy delay line of Eq. (7).
            x_new[1:] = x[:-1]
            x = x_new
            X[n] = x

        return X


class RidgeReadout:
    """Closed-form Tikhonov-regularized least-squares readout."""

    def __init__(self, beta):
        self.beta = float(beta)
        self.coef_ = None

    @staticmethod
    def _augment(X):
        X = np.asarray(X, dtype=float)
        return np.c_[np.ones(len(X)), X]

    def fit(self, X, y):
        A = self._augment(X)
        y = np.asarray(y, dtype=float)
        I = np.eye(A.shape[1], dtype=float)
        I[0, 0] = 0.0  # do not regularize intercept
        self.coef_ = np.linalg.solve(
            A.T @ A + self.beta * I,
            A.T @ y,
        )
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("RidgeReadout must be fitted before prediction.")
        return self._augment(X) @ self.coef_
