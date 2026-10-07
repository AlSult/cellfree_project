# Decides WHERE the APs stand (a jittered grid) and where the users are (random drops).

import numpy as np

import mid_config as cfg


def grid_points():
    """The perfect grid: 5 x 5 points, 100 m apart, starting 50 m from the border."""
    centres = (np.arange(cfg.GRID_SIDE) + 0.5) * cfg.AP_SPACING_M
    xx, yy = np.meshgrid(centres, centres)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def generate_ap_positions(seed=cfg.SEED_LAYOUT):
    """25 APs: the grid, with every AP moved randomly by at most AP_JITTER_M in x and in y."""
    rng = np.random.default_rng(seed)
    grid = grid_points()
    moved = grid + rng.uniform(-cfg.AP_JITTER_M, cfg.AP_JITTER_M, grid.shape)
    return np.clip(moved, 0, cfg.AREA_M)


def generate_user_drop(rng):
    """One drop: every user is placed at a random spot of the map (uniform)."""
    return rng.uniform(0, cfg.AREA_M, (cfg.N_UE, 2))


if __name__ == "__main__":
    print("first 5 APs:\n", generate_ap_positions()[:5].round(1))
    print("a user drop (first 5 users):\n", generate_user_drop(np.random.default_rng(0))[:5].round(1))