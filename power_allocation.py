# Decides WHICH AP serves WHICH user, and HOW MUCH power every beam gets.

import numpy as np

import config


def nearest_ap_mask(dist):
    """Table of True/False with one row per AP and one column per user.
    True means "this AP serves this user".
    Rule used in Step 5: every user is served ONLY by its nearest AP.
    (If two APs are equally near, the one with the lower number wins.)"""
    mask = np.zeros((config.N_AP, config.N_UE), dtype=bool)
    nearest = np.argmin(dist, axis=0)
    mask[nearest, np.arange(config.N_UE)] = True
    return mask


def equal_power_allocation(mask):
    """Every beam gets the same power: the AP's power budget divided by the
    number of users in the whole network. Beams that do not exist get 0.
    Returns p, a table of power in Watts, one row per AP, one column per user."""
    ptx_w = 10 ** ((config.PTX_DBM - 30) / 10)      # 46 dBm = 39.8 W
    p_link = ptx_w / config.N_UE
    return np.where(mask, p_link, 0.0)


if __name__ == "__main__":
    from ap_location_generation import generate_ap_positions
    from ue_location_generation import generate_ue_positions
    from path_loss_model import beta_from_positions
    dist, _, _ = beta_from_positions(generate_ap_positions(), generate_ue_positions())
    mask = nearest_ap_mask(dist)
    print("serving mask:\n", mask)
    print("power (W):\n", equal_power_allocation(mask).round(2))