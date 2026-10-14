# Runs the network for the channel model of the paper (Eq. 2), with
#   - independent STATIC drops (users uniform, or taken from the steady state of Random Waypoint), or
#   - walking EPISODES (Random Waypoint), where the slowly changing part of the channel evolves with the walk.
# The serving rules, the power split and the speed calculation are the ones of paper_serving.py and paper_simulate.py.

import numpy as np

import paper_config as cfg
import paper_layout as layout
import paper_mobility as mob
import paper_models as models
import paper_serving as serving
import paper_simulate as sim
from paper_precoding import mrt_precoders


def evaluate(snap, rules):
    """Speeds of every user under every serving rule, for one snapshot (a dict with beta and H)."""
    out = {}
    for name, rule in rules.items():
        mask = rule(snap["beta"])
        power = serving.equal_power_per_ap(mask)
        signal, interference, sinr, rate = sim.fast_sinr_and_rate(snap["H"], mrt_precoders(snap["H"], power))
        out[name] = {"mask": mask, "rate": rate, "signal": signal, "interference": interference}
    return out


def static_drops(channel_name, ap_pos, rules, n_drops=cfg.N_DROPS, positions="uniform", seed=cfg.SEED_DROPS,
                 n_ant=cfg.N_ANTENNAS, position_fn=None, channel_obj=None, **channel_options):
    """Independent drops. positions: 'uniform' or 'rwp' (a snapshot of the Random Waypoint steady state).
    position_fn(rng) -> (users, 2) overrides positions;  channel_obj is a ready-made channel (instead of channel_name).
    Returns arrays of shape (drops, users) for every rule, in the same layout as paper_simulate.simulate."""
    channel = channel_obj if channel_obj is not None else models.make_channel(channel_name, ap_pos, n_ant, **channel_options)
    K, M = cfg.N_UE, cfg.N_AP
    out = {"dist_nearest": np.zeros((n_drops, K)), "strongest_los": np.zeros((n_drops, K), dtype=bool), "rules": {}}
    for name in rules:
        out["rules"][name] = {key: np.zeros((n_drops, K)) for key in ("rate", "signal", "interference", "n_aps")}
        out["rules"][name]["uncovered"] = 0
        out["rules"][name]["users_per_ap"] = np.zeros((n_drops, M), dtype=int)
    for i in range(n_drops):
        rng = np.random.default_rng([seed, i])
        if position_fn is not None:
            ue = position_fn(rng)
        else:
            ue = layout.generate_user_drop(rng) if positions == "uniform" else mob.stationary_positions(K, rng)
        snap = channel.snapshot(ue, channel.new_state(ue, rng), rng)
        out["dist_nearest"][i] = snap["dist"].min(axis=0)
        if snap["los"] is not None:
            out["strongest_los"][i] = snap["los"][np.argmax(snap["beta"], axis=0), np.arange(K)]
        for name, r in evaluate(snap, rules).items():
            o = out["rules"][name]
            o["rate"][i], o["signal"][i], o["interference"][i] = r["rate"], r["signal"], r["interference"]
            o["n_aps"][i] = r["mask"].sum(axis=0)
            o["uncovered"] += int((~r["mask"].any(axis=0)).sum())
            o["users_per_ap"][i] = r["mask"].sum(axis=1)
    return out


def run_episode(channel, trajectory, rules, rng):
    """One walking episode. trajectory: positions (steps, users, 2). The slowly changing part of the channel
    (shadowing, LOS state, spreads) evolves with the distance every user walked between two snapshots;
    the fast fading is drawn again at every snapshot."""
    T, K, _ = trajectory.shape
    M = len(channel.ap_pos)
    state = channel.new_state(trajectory[0], rng)
    rec = {"rate": {n: np.zeros((T, K)) for n in rules}, "mask": {n: np.zeros((T, M, K), dtype=bool) for n in rules},
           "strongest": np.zeros((T, K), dtype=int), "los_strongest": np.zeros((T, K), dtype=bool)}
    for t in range(T):
        if t > 0:
            moved = np.linalg.norm(trajectory[t] - trajectory[t - 1], axis=1)
            state = channel.step_state(state, moved, rng)
        snap = channel.snapshot(trajectory[t], state, rng)
        rec["strongest"][t] = np.argmax(snap["beta"], axis=0)
        if snap["los"] is not None:
            rec["los_strongest"][t] = snap["los"][rec["strongest"][t], np.arange(K)]
        for name, r in evaluate(snap, rules).items():
            rec["rate"][name][t], rec["mask"][name][t] = r["rate"], r["mask"]
    return rec


def mobility_metrics(rec, trajectory, rule_names):
    """What walking does to the serving decisions and to the speeds.
    changes per km: how often the serving AP (conventional rule) or the set of serving APs (clusters) changes per km walked."""
    T, K, _ = trajectory.shape
    km = np.linalg.norm(np.diff(trajectory, axis=0), axis=2).sum() / 1000.0
    out = {"km_walked": km}
    out["nearest_changes_per_km"] = (np.diff(rec["strongest"], axis=0) != 0).sum() / km
    for name in rule_names:
        mask = rec["mask"][name]
        changed_pairs = (mask[1:] != mask[:-1]).sum(axis=1)                  # (T-1, K): APs that joined or left a user's set
        out[f"{name}: AP joins/leaves per km"] = changed_pairs.sum() / km
        out[f"{name}: share of snapshots with a change"] = float((changed_pairs > 0).mean())
        r = rec["rate"][name]
        out[f"{name}: speed variation over time (std/mean)"] = float((r.std(axis=0) / r.mean(axis=0)).mean())
    return out


if __name__ == "__main__":
    import time
    ap = layout.generate_ap_positions()
    rules = sim.standard_rules()
    for ch in ("eq2",):
        t = time.time()
        res = static_drops(ch, ap, rules, n_drops=5)
        print(ch, {k: round(float(v["rate"].mean())) for k, v in res["rules"].items()}, f"{(time.time() - t) / 5:.2f} s per drop")