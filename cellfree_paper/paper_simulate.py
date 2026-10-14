# One DROP = users placed at random + one random channel. Every serving rule is evaluated on
# the SAME drop, so the comparison between rules is fair. The channel and the aiming come from
# paper_channel.py and paper_precoding.py. The "what does every user hear from every beam" step uses a
# batch of matrix products (fast at this size); it gives the same numbers as the library function
# sinr_and_rate in paper_rates.py (the audit checks it).

import numpy as np

import paper_config as config
import paper_layout as layout
import paper_gains as gains
import paper_serving as serving
from paper_channel import full_channel
from paper_precoding import mrt_precoders
from paper_rates import NOISE_W


def standard_rules(L=config.L_PER_AP):
    return {"nearest AP only": serving.nearest_mask,
            f"clusters (L={L})": lambda beta: serving.cluster_mask(beta, L),
            "everyone serves everyone": serving.everyone_mask}


def fast_sinr_and_rate(H, W, noise_w=NOISE_W):
    """Same result as paper_rates.sinr_and_rate, computed with batched matrix products."""
    A = np.matmul(H.conj(), W.transpose(0, 2, 1))         # A[m, k, j] = what user k hears from AP m's beam for user j
    S = A.sum(axis=0)                                     # all APs add up
    signal = np.abs(np.diag(S)) ** 2
    interference = (np.abs(S) ** 2).sum(axis=1) - signal
    sinr = signal / (interference + noise_w)
    return signal, interference, sinr, config.B_HZ * np.log2(1 + sinr) / 1e6


def simulate_drop(ap_pos, rng, rules, alpha=2.0, shadow_sigma_db=0.0, n_ant=None):
    """Returns everything about one drop (a dictionary)."""
    n_ant = config.N_ANTENNAS if n_ant is None else n_ant
    ue = layout.generate_user_drop(rng)
    dist = gains.distance_matrix(ap_pos, ue)
    beta = gains.beta_from_distance(dist, alpha, shadow_sigma_db, rng)
    H = full_channel(beta, rng, n_ant)                            # ONE random channel for all rules
    drop = {"ue": ue, "dist": dist, "beta": beta, "H": H, "rules": {}}
    for name, rule in rules.items():
        mask = rule(beta)
        power = serving.equal_power_per_ap(mask)
        signal, interference, sinr, rate = fast_sinr_and_rate(H, mrt_precoders(H, power))
        drop["rules"][name] = {"mask": mask, "power": power, "signal": signal, "interference": interference,
                               "sinr": sinr, "rate": rate}
    return drop


def simulate(ap_pos, rules, n_drops=config.N_DROPS, alpha=2.0, shadow_sigma_db=0.0, seed=config.SEED_DROPS, n_ant=None):
    """Many drops. Returns, for every rule, arrays of shape (drops, users)."""
    rng = np.random.default_rng(seed)
    out = {"dist_nearest": np.zeros((n_drops, config.N_UE)), "rules": {}}
    for name in rules:
        out["rules"][name] = {key: np.zeros((n_drops, config.N_UE)) for key in ("rate", "signal", "interference", "n_aps")}
        out["rules"][name]["uncovered"] = 0
        out["rules"][name]["users_per_ap"] = np.zeros((n_drops, config.N_AP), dtype=int)
    for i in range(n_drops):
        drop = simulate_drop(ap_pos, rng, rules, alpha, shadow_sigma_db, n_ant)
        out["dist_nearest"][i] = drop["dist"].min(axis=0)
        for name, r in drop["rules"].items():
            o = out["rules"][name]
            o["rate"][i], o["signal"][i], o["interference"][i] = r["rate"], r["signal"], r["interference"]
            o["n_aps"][i] = r["mask"].sum(axis=0)
            o["uncovered"] += int((~r["mask"].any(axis=0)).sum())
            o["users_per_ap"][i] = r["mask"].sum(axis=1)
    return out


def summarize(rates):
    """The numbers we report for one rule: rates has shape (drops, users)."""
    r = rates.ravel()
    return {"mean_Mbps": r.mean(), "median_Mbps": np.median(r), "p5_Mbps": np.percentile(r, 5),
            "p95_Mbps": np.percentile(r, 95), "sum_per_drop_Mbps": rates.sum(axis=1).mean(),
            "share_below_100Mbps_pct": 100 * np.mean(r < 100)}


if __name__ == "__main__":
    ap = layout.generate_ap_positions()
    res = simulate(ap, standard_rules(), n_drops=20)
    for name, o in res["rules"].items():
        print(f"{name:26s} mean speed {o['rate'].mean():6.0f} Mbps")