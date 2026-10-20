import numpy as np

from .physics import build_timing_templates, reduced_log_likelihood
from .reservoir import PhysicalDelayLineReservoir, RidgeReadout


def _resample_rows(X, n_rows):
    """Deterministically map any trajectory length to exactly n_rows."""
    X = np.asarray(X, dtype=float)
    if len(X) == n_rows:
        return X
    idx = np.linspace(0, len(X) - 1, n_rows).round().astype(int)
    return X[idx]


class RCPHY:
    """Physics module: fixed physical reservoir + ridge likelihood readout."""

    def __init__(self, cfg, bits):
        self.cfg = cfg
        self.bits = np.asarray(bits, dtype=int)
        self.reservoir = PhysicalDelayLineReservoir(cfg)
        self.readout = RidgeReadout(cfg.ridge_beta)
        self.templates = build_timing_templates(self.bits, cfg)

    def _candidate_states(self, y):
        """Create J states indexed by the timing grid.

        The reservoir processes the observation once; the state trajectory is
        deterministically aligned with the J timing hypotheses. This is the
        paper's Eq. (12) view in which readout index p corresponds to
        Delta t = p * t_s.
        """
        X = self.reservoir.transform(y)
        return _resample_rows(X, self.cfg.J)

    def fit(self, samples):
        X_all, y_all = [], []
        for sample in samples:
            Xj = self._candidate_states(sample["y"])
            target = reduced_log_likelihood(
                sample["y"],
                self.templates,
                self.cfg,
            )
            X_all.append(Xj)
            y_all.append(target)

        self.readout.fit(np.vstack(X_all), np.concatenate(y_all))
        return self

    def score_landscape(self, y):
        return self.readout.predict(self._candidate_states(y))

    def estimate_argmax(self, y):
        j = int(np.argmax(self.score_landscape(y)))
        return float(self.cfg.delta_grid[j])


class RCAGENT:
    """Agent module aligned with Eqs. (13)–(14).

    Input: RC-PHY reward/likelihood landscape.
    Fixed dynamics: same physical reservoir as RC-PHY.
    Trainable part: one ridge readout.
    Output: timing estimate, snapped to the finite action set D.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.reservoir = PhysicalDelayLineReservoir(cfg)
        self.readout = RidgeReadout(cfg.ridge_beta)

    @staticmethod
    def _standardize_reward(reward):
        reward = np.asarray(reward, dtype=float)
        return (reward - reward.mean()) / (reward.std() + 1e-12)

    def _final_state(self, reward_landscape):
        r = self._standardize_reward(reward_landscape)
        X = self.reservoir.transform(r)
        return X[-1]

    def fit(self, samples, rc_phy):
        X = np.asarray(
            [self._final_state(rc_phy.score_landscape(s["y"])) for s in samples],
            dtype=float,
        )
        y = np.asarray([s["delta_true"] for s in samples], dtype=float)
        self.readout.fit(X, y)
        return self

    def estimate(self, y, rc_phy):
        state = self._final_state(rc_phy.score_landscape(y))[None, :]
        raw = float(self.readout.predict(state)[0])
        raw = float(np.clip(raw, 0.0, self.cfg.T - self.cfg.delta_step))
        j = int(np.argmin(np.abs(self.cfg.delta_grid - raw)))
        return float(self.cfg.delta_grid[j])


class DualRCSystem:
    def __init__(self, cfg, bits=None):
        self.cfg = cfg
        self.bits = cfg.synch_frame if bits is None else np.asarray(bits, dtype=int)
        self.rc_phy = RCPHY(cfg, self.bits)
        self.rc_agent = RCAGENT(cfg)

    def fit(self, train_samples):
        self.rc_phy.fit(train_samples)
        self.rc_agent.fit(train_samples, self.rc_phy)
        return self

    def estimate(self, y):
        return self.rc_agent.estimate(y, self.rc_phy)
