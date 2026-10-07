# One DROP = users placed at random + one random channel. Every serving rule is evaluated on
# the SAME drop, so the comparison between rules is fair. The physics is not rewritten here:
# channel, aiming and speed come from the files of Steps 3 to 5, which work for any size.

import numpy as np

import mid_config as cfg
import mid_layout
import mid_gains
import mid_serving
from channel_model import full_channel
from precoding import mrt_precoders
from rate_calculation import sinr_and_rate, NOISE_W


def standard_rules(L=cfg.L_PER_AP):
    return {"nearest AP only": mid_serving.nearest_mask,
            f"clusters (L={L})": lambda beta: mid_serving.cluster_mask(beta, L),
            "everyone serves everyone": mid_serving.everyone_mask}


def simulate_drop(ap_pos, rng, rules, alpha=2.0, shadow_sigma_db=0.0):
    """Returns everything about one drop (a dictionary)."""
    ue = mid_layout.generate_user_drop(rng)
    dist = mid_gains.distance_matrix(ap_pos, ue)
    beta = mid_gains.beta_from_distance(dist, alpha, shadow_sigma_db, rng)
    H = full_channel(beta, rng)                                   # ONE random channel for all rules
    drop = {"ue": ue, "dist": dist, "beta": beta, "H": H, "rules": {}}
    for name, rule in rules.items():
        mask = rule(beta)
        power = mid_serving.equal_power_per_ap(mask)
        signal, interference, sinr, rate = sinr_and_rate(H, mrt_precoders(H, power))
        drop["rules"][name] = {"mask": mask, "power": power, "signal": signal, "interference": interference,
                               "sinr": sinr, "rate": rate}
    return drop


def simulate(ap_pos, rules, n_drops=cfg.N_DROPS, alpha=2.0, shadow_sigma_db=0.0, seed=cfg.SEED_DROPS):
    """Many drops. Returns, for every rule, arrays of shape (drops, users)."""
    rng = np.random.default_rng(seed)
    out = {"dist_nearest": np.zeros((n_drops, cfg.N_UE)), "rules": {}}
    for name in rules:
        out["rules"][name] = {key: np.zeros((n_drops, cfg.N_UE)) for key in ("rate", "signal", "interference", "n_aps")}
        out["rules"][name]["uncovered"] = 0
        out["rules"][name]["users_per_ap"] = np.zeros((n_drops, cfg.N_AP), dtype=int)
    for i in range(n_drops):
        drop = simulate_drop(ap_pos, rng, rules, alpha, shadow_sigma_db)
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
    ap = mid_layout.generate_ap_positions()
    res = simulate(ap, standard_rules(), n_drops=20)
    for name, o in res["rules"].items():
        print(f"{name:26s} mean speed {o['rate'].mean():6.0f} Mbps")