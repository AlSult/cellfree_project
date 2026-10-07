# WHICH APs serve WHICH user, and HOW MUCH power every beam gets.
# A "mask" is a table of True/False: one row per AP, one column per user.

import numpy as np

import mid_config as cfg


def nearest_mask(beta):
    """Conventional rule: every user is served ONLY by the AP it hears best (the nearest one)."""
    mask = np.zeros(beta.shape, dtype=bool)
    mask[np.argmax(beta, axis=0), np.arange(beta.shape[1])] = True
    return mask


def everyone_mask(beta):
    """Full cooperation: every AP serves every user."""
    return np.ones(beta.shape, dtype=bool)


def cluster_mask(beta, L):
    """User-centric cell-free rule (the new one):
       1. every AP picks its L strongest users;
       2. a user that nobody picked is handed to the AP that hears it best; that AP drops its
          weakest user, but only a user that is still served by some other AP.
    So every AP serves exactly L users and every user is served by at least one AP."""
    n_ap, n_ue = beta.shape
    mask = np.zeros((n_ap, n_ue), dtype=bool)
    strongest = np.argsort(-beta, axis=1)[:, :L]
    mask[np.arange(n_ap)[:, None], strongest] = True
    for user in np.where(~mask.any(axis=0))[0]:
        for ap in np.argsort(-beta[:, user]):                       # best AP first
            droppable = np.where(mask[ap] & (mask.sum(axis=0) > 1))[0]
            if len(droppable):
                drop = droppable[np.argmin(beta[ap, droppable])]
                mask[ap, drop] = False
                mask[ap, user] = True
                break
    return mask


def equal_power_per_ap(mask):
    """Every AP that sends something uses its WHOLE power budget, split equally between its beams.
    An AP that serves nobody stays silent. Returns the power of every beam in Watts."""
    budget_w = 10 ** ((cfg.PTX_DBM - 30) / 10)
    beams_per_ap = mask.sum(axis=1, keepdims=True)
    return np.where(mask, budget_w / np.maximum(beams_per_ap, 1), 0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    beta = rng.random((cfg.N_AP, cfg.N_UE)) ** 6
    for name, mask in [("nearest", nearest_mask(beta)), ("cluster", cluster_mask(beta, 3)), ("everyone", everyone_mask(beta))]:
        print(f"{name:9s} beams {mask.sum():3d}   users per AP {mask.sum(axis=1).min()}..{mask.sum(axis=1).max()}   "
              f"every user served: {bool(mask.any(axis=0).all())}")