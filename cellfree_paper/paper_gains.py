# Distances and channel gains (beta) for every AP-user pair.

import numpy as np

import paper_config as config


def distance_matrix(ap_pos, ue_pos):
    """Distance in metres from every AP (rows) to every user (columns)."""
    return np.linalg.norm(ap_pos[:, None, :] - ue_pos[None, :, :], axis=2)


def beta_from_distance(dist, alpha=2.0, shadow_sigma_db=0.0, rng=None):
    """Fraction of power that survives (beta) for every AP-user pair.

    loss in dB = (loss at 1 m) + 10 x alpha x log10(distance) + shadowing
      alpha = 2    : free space
      alpha = 3.8  : the path loss exponent of Table I of the paper
      shadowing    : a random extra loss, 'normal' in dB with spread shadow_sigma_db
                     (a different random number for every AP-user pair)
    Distances are never taken below 1 m (a user standing on an AP)."""
    d = np.maximum(dist, 1.0)
    loss_at_1m_db = 20 * np.log10(4 * np.pi * config.FC_HZ / config.C)
    loss_db = loss_at_1m_db + 10 * alpha * np.log10(d)
    if shadow_sigma_db > 0:
        loss_db = loss_db + rng.normal(0.0, shadow_sigma_db, d.shape)
    return 10 ** (-loss_db / 10)