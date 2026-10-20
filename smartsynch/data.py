import numpy as np
from .physics import generate_observation


def balanced_delta_indices(n, cfg, rng):
    ids = np.tile(np.arange(cfg.J), int(np.ceil(n / cfg.J)))[:n]
    rng.shuffle(ids)
    return ids


def generate_positive_dataset(n, bits, cfg, rng):
    data = []
    for j in balanced_delta_indices(n, cfg, rng):
        dt = float(cfg.delta_grid[int(j)])
        y = generate_observation(bits, dt, cfg, rng)
        data.append(
            {
                "y": y,
                "delta_true": dt,
                "delta_idx": int(j),
            }
        )
    return data
