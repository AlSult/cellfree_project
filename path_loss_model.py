# Answers ONE question: "how much of the signal survives over a distance?"

import math
import numpy as np

import config


def distance_table(ap_pos, ue_pos):
    """Distance in metres from every AP (rows) to every user (columns)."""
    return np.linalg.norm(ap_pos[:, None, :] - ue_pos[None, :, :], axis=2)


def free_space_path_loss_db(d_m):
    """Signal loss in dB after d_m metres (free space, Friis formula)."""
    return 20 * math.log10(4 * math.pi * d_m * config.FC_HZ / config.C)


def beta_from_positions(ap_pos, ue_pos):
    """For every AP-user pair returns three tables:
       dist  : distance in metres
       pl_db : path loss in dB (bigger number = weaker signal)
       beta  : the fraction of signal power that survives (between 0 and 1)
    """
    dist = distance_table(ap_pos, ue_pos)
    pl_db = np.array([[free_space_path_loss_db(d) for d in row] for row in dist])
    beta = 10 ** (-pl_db / 10)
    return dist, pl_db, beta


if __name__ == "__main__":
    from ap_location_generation import generate_ap_positions
    from ue_location_generation import generate_ue_positions
    dist, pl_db, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
    print("path loss (dB):\n", pl_db.round(1))