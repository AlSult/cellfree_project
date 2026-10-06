# A quick estimate of every user's speed that needs NO random numbers.
# It is the "paper and pencil" rule of Steps 5 and 6, written once as a function:
#     wanted signal = (sum over the APs of sqrt(power x antennas x beta))^2
#     interference  = sum over the other beams j and over the APs of  power x beta
# We use it as an independent reference to check the random simulation.

import numpy as np

import config
from rate_calculation import NOISE_W


def expected_rate(beta, p):
    """beta : (N_AP, N_UE) channel gains    p : (N_AP, N_UE) power of every beam (W)
    Returns the estimated speed of every user in Mbps."""
    n_ue = beta.shape[1]
    signal = np.sum(np.sqrt(p * config.N_ANTENNAS * beta), axis=0) ** 2
    interference = np.zeros(n_ue)
    for k in range(n_ue):
        for j in range(n_ue):
            if j != k:
                interference[k] += np.sum(p[:, j] * beta[:, k])
    sinr = signal / (interference + NOISE_W)
    return config.B_HZ * np.log2(1 + sinr) / 1e6


if __name__ == "__main__":
    from ap_location_generation import generate_ap_positions
    from ue_location_generation import generate_ue_positions
    from path_loss_model import beta_from_positions
    from power_allocation import nearest_ap_mask, cooperative_mask, equal_power_allocation
    dist, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
    for name, mask in [("Step 5 (nearest AP)", nearest_ap_mask(dist)), ("Step 6 (cell-free)", cooperative_mask())]:
        print(name, expected_rate(beta, equal_power_allocation(mask)).round(0), "Mbps")