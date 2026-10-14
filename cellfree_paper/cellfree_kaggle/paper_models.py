#(only the channel written in the paper, Eq. 1 and 2)
#
#   Eq2Channel   h = sqrt(beta) * g,   beta = beta0 * d^-alpha * 10^(zeta/10),   g ~ CN(0, I)  (independent Rayleigh fading)
#
# Interface (the same for every channel we may add later):
#   channel.new_state(ue_pos, rng)              random numbers that describe the slowly changing part (the shadowing)
#   channel.step_state(state, moved_m, rng)     let the slow part evolve while the users walked moved_m metres
#   channel.snapshot(ue_pos, state, rng)        -> dict with dist (2D distance), beta, H (the channel), los
#
# The shadowing of one AP-user link is remembered while the user walks (a first-order autoregressive process):
#   z_new = rho * z_old + sqrt(1 - rho^2) * fresh,   rho = exp(-moved / corr_dist)       (corr_dist = 50 m, our assumption)
# The fast fading g is drawn again at every snapshot (at 3.5 GHz it is uncorrelated after about a millisecond).

import numpy as np

import paper_config as cfg
import paper_gains as gains
from paper_channel import full_channel


class Eq2Channel:
    name = "eq2"

    def __init__(self, ap_pos, n_ant=cfg.N_ANTENNAS, alpha=cfg.PAPER_LIKE["alpha"], sigma_db=cfg.PAPER_LIKE["shadow_sigma_db"],
                 corr_dist_m=cfg.SHADOW_CORR_DIST_EQ2_M):
        self.ap_pos, self.n_ant, self.alpha, self.sigma_db, self.corr = ap_pos, n_ant, alpha, sigma_db, corr_dist_m

    def new_state(self, ue_pos, rng):
        return {"z_sf": rng.standard_normal((len(self.ap_pos), len(ue_pos)))}

    def step_state(self, state, moved_m, rng):
        rho = np.exp(-np.asarray(moved_m) / self.corr)[None, :]
        return {"z_sf": rho * state["z_sf"] + np.sqrt(1 - rho ** 2) * rng.standard_normal(state["z_sf"].shape)}

    def snapshot(self, ue_pos, state, rng):
        dist = gains.distance_matrix(self.ap_pos, ue_pos)
        beta = gains.beta_from_distance(dist, self.alpha) * 10 ** (-self.sigma_db * state["z_sf"] / 10.0)
        return {"dist": dist, "beta": beta, "H": full_channel(beta, rng, self.n_ant), "los": None}


def make_channel(name, ap_pos, n_ant=cfg.N_ANTENNAS, **options):
    return {"eq2": Eq2Channel}[name](ap_pos, n_ant, **options)


if __name__ == "__main__":
    import paper_layout as layout
    ap = layout.generate_ap_positions()
    ch = Eq2Channel(ap)
    rng = np.random.default_rng(0)
    ue = layout.generate_user_drop(rng)
    snap = ch.snapshot(ue, ch.new_state(ue, rng), rng)
    print("channel", snap["H"].shape, " beta range (dB): %.0f ... %.0f" % tuple(10 * np.log10([snap["beta"].min(), snap["beta"].max()])))