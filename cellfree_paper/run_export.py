# How do we SAVE the paper-size network? Saving the full channel is impossible (12.4 MB per snapshot,
# 12 GB for 1000 snapshots). Instead we save the large-scale gains (beta), the serving table and the
# speeds, and we store the SEED of every snapshot: with the seed, the random channel of any snapshot can
# be recomputed exactly whenever it is needed.
# Output: data/snapshots.npz + data/snapshots_info.json

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the files of THIS folder always win
import json
import time
import numpy as np

import paper_config as cfg
import paper_layout as layout
import paper_serving as serving
import paper_simulate as sim
import paper_paths as paths

N_SNAPSHOTS = 200
SNAPSHOT_FILE = paths.data("snapshots.npz")
BASE_SEED = 20261007            # snapshot number i uses the random generator seeded with [BASE_SEED, i]
PROPAGATION = cfg.PAPER_LIKE
RULES = sim.standard_rules()
CLUSTER = [name for name in RULES if "cluster" in name][0]


def snapshot(ap_pos, index):
    """One snapshot, completely decided by (BASE_SEED, index): users, shadowing, channel."""
    return sim.simulate_drop(ap_pos, np.random.default_rng([BASE_SEED, index]), RULES, **PROPAGATION)


def build(n_snapshots=N_SNAPSHOTS):
    ap_pos = layout.generate_ap_positions()
    M, K, L = cfg.N_AP, cfg.N_UE, cfg.L_PER_AP
    a = {"ap_pos": ap_pos.astype(np.float32),
         "ue_pos": np.zeros((n_snapshots, K, 2), np.float32),
         "beta": np.zeros((n_snapshots, M, K), np.float32),
         "cluster_users": np.zeros((n_snapshots, M, L), np.int16)}
    for name in RULES:
        a["rate_" + name.split(" ")[0]] = np.zeros((n_snapshots, K), np.float32)
    for i in range(n_snapshots):
        d = snapshot(ap_pos, i)
        a["ue_pos"][i], a["beta"][i] = d["ue"], d["beta"]
        mask = d["rules"][CLUSTER]["mask"]
        for m in range(M):                                   # the L users of every AP, strongest first
            users = np.where(mask[m])[0]
            a["cluster_users"][i, m] = users[np.argsort(-d["beta"][m, users])]
        for name, r in d["rules"].items():
            a["rate_" + name.split(" ")[0]][i] = r["rate"]
    return a


if __name__ == "__main__":
    start = time.perf_counter()
    arrays = build()
    seconds = time.perf_counter() - start
    np.savez_compressed(SNAPSHOT_FILE, **arrays)
    info = {"description": "paper-size cell-free network snapshots (121 APs, 100 users, 64 antennas, paper-like propagation)",
            "snapshots": N_SNAPSHOTS, "base_seed": BASE_SEED,
            "seed_of_snapshot_i": "numpy.random.default_rng([base_seed, i])", "propagation": PROPAGATION,
            "n_ap": cfg.N_AP, "n_ue": cfg.N_UE, "n_antennas": cfg.N_ANTENNAS, "area_m": cfg.AREA_M,
            "ap_jitter_m": cfg.AP_JITTER_M, "layout_seed": cfg.SEED_LAYOUT, "users_per_ap_L": cfg.L_PER_AP,
            "rules": list(RULES), "units": {"ue_pos": "m", "beta": "linear power gain", "rate_*": "Mbps",
                                            "cluster_users": "user numbers served by every AP, strongest first"},
            "not_stored": "the random channel H (12.4 MB per snapshot); it is recomputed from the seed when needed"}
    with open(paths.data("snapshots_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    size = os.path.getsize(SNAPSHOT_FILE)
    per_snapshot = sum(v.nbytes for k, v in arrays.items() if k != "ap_pos") / N_SNAPSHOTS
    channel_bytes = cfg.N_AP * cfg.N_UE * cfg.N_ANTENNAS * 16
    print(f"{N_SNAPSHOTS} snapshots built in {seconds:.0f} s ({seconds / N_SNAPSHOTS:.2f} s each)\n")
    print(f"{'what':28s} {'shape':>16} {'type':>8} {'bytes per snapshot':>20}")
    for k, v in arrays.items():
        if k != "ap_pos":
            print(f"{k:28s} {str(v.shape[1:]):>16} {str(v.dtype):>8} {v[0].nbytes:20,d}")
    print(f"{'TOTAL':28s} {'':>16} {'':>8} {per_snapshot:20,.0f}")
    print(f"{'the channel H (not stored)':28s} {str((cfg.N_AP, cfg.N_UE, cfg.N_ANTENNAS)):>16} {'complex':>8} {channel_bytes:20,d}\n")
    print(f"{'':28s} {'what we store':>16} {'with the channel':>18}")
    for n in (200, 1000, 10000):
        print(f"{n:6d} snapshots {'':>13} {n * per_snapshot / 1e6:13.0f} MB {n * channel_bytes / 1e9:15.1f} GB")
    print(f"\nfile on disk for {N_SNAPSHOTS} snapshots: {size / 1e6:.1f} MB\n")

    # ---------------- CHECKS ----------------
    print("CHECKS")
    loaded = np.load(SNAPSHOT_FILE)
    ok1 = all(np.array_equal(loaded[k], arrays[k]) for k in arrays)
    print(f"[1] the saved file reads back exactly as it was written -> {'PASS' if ok1 else 'FAIL'}")
    ap_pos = layout.generate_ap_positions()
    ok2 = True
    for i in (0, 57, 199):                                    # NOT in order: any snapshot can be rebuilt on its own
        d = snapshot(ap_pos, i)
        for name, r in d["rules"].items():
            ok2 &= bool(np.allclose(r["rate"], loaded["rate_" + name.split(" ")[0]][i], rtol=1e-6))
    print(f"[2] snapshots 0, 57 and 199 rebuilt from their seed alone give the stored speeds, so the channel need not be stored -> {'PASS' if ok2 else 'FAIL'}")
    same_table = 0
    for i in range(20):                                       # the first 20 snapshots, all must match exactly
        mask = serving.cluster_mask(loaded["beta"][i].astype(np.float64), cfg.L_PER_AP)
        stored = np.zeros_like(mask)
        for m in range(cfg.N_AP):
            stored[m, loaded["cluster_users"][i, m]] = True
        same_table += int(np.array_equal(mask, stored))
    ok3 = same_table == 20
    print(f"[3] the serving table rebuilt from the stored beta equals the stored one in {same_table} of 20 snapshots -> {'PASS' if ok3 else 'FAIL'}")
    ok4 = bool(np.all(np.isfinite(loaded["beta"])) and np.all(loaded["beta"] > 0) and np.all(loaded["rate_clusters"] > 0))
    print(f"[4] every stored number is finite, and every gain and speed is positive -> {'PASS' if ok4 else 'FAIL'}")
    ok5 = per_snapshot < channel_bytes / 100
    print(f"[5] what we store is more than 100 times smaller than the channel ({channel_bytes / per_snapshot:.0f} times) -> {'PASS' if ok5 else 'FAIL'}")
    print("\nSaved data/snapshots.npz and data/snapshots_info.json")