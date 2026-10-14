# Builds the radio channel: the slow part (path loss, from Step 2)
# multiplied by the fast random part (fading).

import numpy as np

import paper_config as config


def rayleigh_fading(n_ap, n_ue, n_ant, rng):
    """Random numbers g with an average power of exactly 1.
    Every antenna of every AP-user pair gets its own complex number
    (a real part and an imaginary part, each random)."""
    real = rng.standard_normal((n_ap, n_ue, n_ant))
    imag = rng.standard_normal((n_ap, n_ue, n_ant))
    return (real + 1j * imag) / np.sqrt(2)


def full_channel(beta, rng, n_ant=config.N_ANTENNAS):
    """The channel H = sqrt(beta) * g.
    beta : (N_AP, N_UE) table from Step 2 (fraction of power that survives)
    H    : (N_AP, N_UE, n_ant) table of complex numbers, one per antenna
    """
    n_ap, n_ue = beta.shape
    g = rayleigh_fading(n_ap, n_ue, n_ant, rng)
    return np.sqrt(beta)[:, :, None] * g


if __name__ == "__main__":
    import paper_layout as layout
    import paper_gains as gains
    ap = layout.generate_ap_positions()
    ue = layout.generate_user_drop(np.random.default_rng(config.SEED_DROPS))
    beta = gains.beta_from_distance(gains.distance_matrix(ap, ue))
    H = full_channel(beta, np.random.default_rng(config.SEED_DROPS))
    print("H shape:", H.shape)