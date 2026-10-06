# Decides HOW an AP uses its 16 antennas to send a signal to a user.
# Precoding = giving every antenna its own "recipe" (a number) for the signal.

import numpy as np


def mrt_precoders(H, p):
    """Aim the antennas at a user (maximum-ratio transmission, MRT).
    Every antenna sends the user's signal with a weight that matches that
    antenna's channel to the user, so all 16 copies arrive lined up.
        H : (N_AP, N_UE, N_ANT) channel numbers from Step 3
        p : (N_AP, N_UE) power the AP uses for each user
        W : (N_AP, N_UE, N_ANT) the weights (one per antenna)
    The weights always use exactly the power p (their squared length is p)."""
    direction = H / np.linalg.norm(H, axis=2, keepdims=True)
    return np.sqrt(p)[:, :, None] * direction


def no_aim_precoders(H, p):
    """For comparison: NO aiming. Every antenna sends the same thing with
    the same weight, like a lamp that lights up the whole room."""
    n_ap, n_ue, n_ant = H.shape
    return np.sqrt(p)[:, :, None] * np.ones((n_ap, n_ue, n_ant)) / np.sqrt(n_ant)


def signal_seen_by_users(H, W):
    """A[m, k, j] = the signal that user k receives from AP m when AP m
    sends the beam that was meant for user j.
        j == k : the wanted signal (the beam aimed at this very user)
        j != k : leakage (a beam aimed at somebody else)
    """
    return np.einsum('mkn,mjn->mkj', H.conj(), W)


if __name__ == "__main__":
    import config
    from ap_location_generation import generate_ap_positions
    from ue_location_generation import generate_ue_positions
    from path_loss_model import beta_from_positions
    from channel_model import full_channel
    _, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
    H = full_channel(beta, np.random.default_rng(config.SEED))
    W = mrt_precoders(H, np.ones(beta.shape))
    print("W shape:", W.shape)