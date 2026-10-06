# Saves the tiny network (2 APs, 4 users, 16 antennas) as ONE data file, so that
# somebody else can check it. Output: tiny_dataset.npz + tiny_dataset_info.json

import json
import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from mobility_model import generate_trajectories
from path_loss_model import beta_from_positions
from channel_model import full_channel
from power_allocation import nearest_ap_mask, cooperative_mask, equal_power_allocation
from precoding import mrt_precoders
from rate_calculation import sinr_and_rate, noise_power_dbm

EPISODES = 10          # 10 independent walks of 40 seconds
WALK_SEED0 = 100       # episode e uses walk seed  WALK_SEED0 + e
CHANNEL_SEED0 = 200    # episode e uses channel seed CHANNEL_SEED0 + e
RULES = ("nearest", "cellfree")


def build_dataset(episodes=EPISODES, walk_seed0=WALK_SEED0, channel_seed0=CHANNEL_SEED0):
    """Returns (arrays, info). Nothing is saved here, so the audit can call it again."""
    ap_pos, ue_start = generate_ap_positions(), generate_ue_positions()
    E, T, M, K, N = episodes, config.N_STEPS, config.N_AP, config.N_UE, config.N_ANTENNAS
    a = {"ap_pos": ap_pos,
         "pos": np.zeros((E, T, K, 2)), "speed": np.zeros((E, K)),
         "dist": np.zeros((E, T, M, K)), "beta": np.zeros((E, T, M, K)),
         "H": np.zeros((E, T, M, K, N), dtype=complex)}
    for r in RULES:
        a[f"mask_{r}"] = np.zeros((E, T, M, K), dtype=bool)
        a[f"power_{r}"] = np.zeros((E, T, M, K))
        for name in ("signal", "interference", "sinr", "rate"):
            a[f"{name}_{r}"] = np.zeros((E, T, K))

    for e in range(episodes):
        pos, speed = generate_trajectories(ue_start, seed=walk_seed0 + e)
        rng = np.random.default_rng(channel_seed0 + e)
        a["pos"][e], a["speed"][e] = pos, speed
        for t in range(T):
            dist, _, beta = beta_from_positions(ap_pos, pos[t])
            H = full_channel(beta, rng)                       # a NEW random channel at every second
            a["dist"][e, t], a["beta"][e, t], a["H"][e, t] = dist, beta, H
            for r in RULES:
                mask = nearest_ap_mask(dist) if r == "nearest" else cooperative_mask()
                p = equal_power_allocation(mask)
                sig, itf, sinr, rate = sinr_and_rate(H, mrt_precoders(H, p))
                a[f"mask_{r}"][e, t], a[f"power_{r}"][e, t] = mask, p
                a[f"signal_{r}"][e, t], a[f"interference_{r}"][e, t] = sig, itf
                a[f"sinr_{r}"][e, t], a[f"rate_{r}"][e, t] = sinr, rate

    info = {
        "description": "tiny cell-free network, 2 APs, 4 users, free-space path loss + Rayleigh fading, MRT aiming",
        "episodes": E, "steps": T, "n_ap": M, "n_ue": K, "n_antennas": N,
        "area_m": config.AREA_M, "fc_hz": config.FC_HZ, "c": config.C,
        "ptx_dbm": config.PTX_DBM, "b_hz": config.B_HZ, "nf_db": config.NF_DB, "n0_dbm_hz": config.N0_DBM_HZ,
        "dt_s": config.DT_S, "speed_min_ms": config.SPEED_MIN_MS, "speed_max_ms": config.SPEED_MAX_MS,
        "noise_dbm": noise_power_dbm(), "walk_seed0": walk_seed0, "channel_seed0": channel_seed0,
        "ue_start": ue_start.tolist(), "rules": list(RULES),
        "units": {"ap_pos": "m", "pos": "m", "speed": "m/s", "dist": "m", "beta": "linear power gain",
                  "H": "complex channel, sqrt(power gain)", "power": "W", "signal": "W", "interference": "W",
                  "sinr": "linear ratio", "rate": "Mbps"},
        "array_axes": {"pos": "episode, time, user, xy", "dist/beta": "episode, time, AP, user",
                       "H": "episode, time, AP, user, antenna", "mask/power": "episode, time, AP, user",
                       "signal/interference/sinr/rate": "episode, time, user"},
        "channel_draw": "independent at every time step",
    }
    return a, info


if __name__ == "__main__":
    arrays, info = build_dataset()
    np.savez_compressed("tiny_dataset/tiny_dataset.npz", **arrays)
    with open("tiny_dataset/tiny_dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved tiny_dataset.npz ({len(arrays)} arrays) and tiny_dataset_info.json")
    for name, arr in arrays.items():
        print(f"   {name:22s} shape {str(arr.shape):22s} dtype {arr.dtype}")