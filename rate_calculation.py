# Turns "what the users receive" into SINR and into a speed in Mbps.

import math
import numpy as np

import config
from precoding import signal_seen_by_users


def noise_power_dbm():
    """Noise floor of the receiver: thermal noise (-174 dBm per Hz) over the
    whole bandwidth, plus the noise added by the receiver itself."""
    return config.N0_DBM_HZ + 10 * math.log10(config.B_HZ) + config.NF_DB


NOISE_W = 10 ** ((noise_power_dbm() - 30) / 10)     # the same noise, in Watts


def sinr_and_rate(H, W, noise_w=NOISE_W):
    """For every user k:
         signal       = power of the beam aimed at k, as k receives it
         interference = power of all the other beams, as k receives them
         SINR         = signal / (interference + noise)
         rate         = bandwidth x log2(1 + SINR)     (Shannon), in Mbps
    H : channel (N_AP, N_UE, N_ANT)     W : weights (N_AP, N_UE, N_ANT)
    """
    A = signal_seen_by_users(H, W)          # A[m, k, j]
    S = A.sum(axis=0)                       # add the contributions of all APs
    signal = np.abs(np.diag(S)) ** 2
    interference = (np.abs(S) ** 2).sum(axis=1) - signal
    sinr = signal / (interference + noise_w)
    rate_mbps = config.B_HZ * np.log2(1 + sinr) / 1e6
    return signal, interference, sinr, rate_mbps


def interference_table(H, W):
    """Table I[k, j] = power that user k receives from the beam meant for
    user j (diagonal set to 0, because that is the wanted signal)."""
    S = signal_seen_by_users(H, W).sum(axis=0)
    table = np.abs(S) ** 2
    np.fill_diagonal(table, 0.0)
    return table


if __name__ == "__main__":
    print(f"noise floor: {noise_power_dbm():.1f} dBm")