# The middle network (25 APs, 20 users): speeds under three serving rules, two kinds of
# propagation, and how many users an AP should serve. Everything is saved as tables.

import csv
import numpy as np

import config
import mid_config as cfg
import mid_layout
import mid_serving
import mid_simulate as sim
from rate_calculation import NOISE_W

ap_pos = mid_layout.generate_ap_positions()
SETTINGS = {"free space (alpha 2, no shadowing)": cfg.FREE_SPACE,
            "paper-like (alpha 3.8, shadowing 8 dB)": cfg.PAPER_LIKE}

print(f"THE MIDDLE NETWORK: {cfg.N_AP} APs on a jittered {cfg.GRID_SIDE} x {cfg.GRID_SIDE} grid ({cfg.AP_SPACING_M:.0f} m apart), "
      f"{cfg.N_UE} users, {cfg.AREA_M:.0f} m x {cfg.AREA_M:.0f} m, {config.N_ANTENNAS} antennas per AP")
print(f"{cfg.N_DROPS} drops (users at random places + one random channel); every rule sees exactly the same drops")
print(f"users per AP on average: {cfg.N_UE / cfg.N_AP:.2f}   (the paper: 100 / 121 = {100 / 121:.2f})\n")

# =====================================================================
# TABLE 1: the three rules, two kinds of propagation
# =====================================================================
header1 = ["propagation", "rule", "mean_Mbps", "median_Mbps", "p5_Mbps", "p95_Mbps", "sum_per_drop_Mbps",
           "share_below_100Mbps_pct", "APs_per_user", "silent_APs_pct"]
rows1, results = [], {}
for sname, s in SETTINGS.items():
    res = sim.simulate(ap_pos, sim.standard_rules(), **s)
    results[sname] = res
    for rname, o in res["rules"].items():
        st = sim.summarize(o["rate"])
        rows1.append([sname, rname, st["mean_Mbps"], st["median_Mbps"], st["p5_Mbps"], st["p95_Mbps"],
                      st["sum_per_drop_Mbps"], st["share_below_100Mbps_pct"], o["n_aps"].mean(),
                      100 * np.mean(o["users_per_ap"] == 0)])
print("TABLE 1: speed per user (Mbps)   p5 = the slowest 5% of the users are below this, p95 = the fastest 5% are above")
print(f"{'rule':27s} {'mean':>6} {'median':>7} {'p5':>6} {'p95':>6} {'sum/drop':>9} {'<100':>6} {'APs/user':>9} {'silent APs':>11}")
for sname in SETTINGS:
    print(f"--- {sname}")
    for r in [r for r in rows1 if r[0] == sname]:
        print(f"{r[1]:27s} {r[2]:6.0f} {r[3]:7.0f} {r[4]:6.0f} {r[5]:6.0f} {r[6]:9.0f} {r[7]:5.1f}% {r[8]:9.2f} {r[9]:10.1f}%")

# =====================================================================
# TABLE 2: how many users should an AP serve? (L)
# =====================================================================
header2 = ["propagation", "L", "mean_Mbps", "p5_Mbps", "sum_per_drop_Mbps", "APs_per_user", "users_not_served"]
rows2 = []
L_VALUES = [1, 2, 3, 4, 6, 10, 20]
for sname, s in SETTINGS.items():
    for L in L_VALUES:
        rules = {"c": (lambda beta, L=L: mid_serving.cluster_mask(beta, L))}
        o = sim.simulate(ap_pos, rules, **s)["rules"]["c"]
        st = sim.summarize(o["rate"])
        rows2.append([sname, L, st["mean_Mbps"], st["p5_Mbps"], st["sum_per_drop_Mbps"], o["n_aps"].mean(), o["uncovered"]])
print("\nTABLE 2: the number of users each AP serves (L). L = 20 means everybody serves everybody")
print(f"{'':39s} {'L':>3} {'mean':>6} {'p5':>6} {'sum/drop':>9} {'APs/user':>9} {'not served':>11}")
for sname in SETTINGS:
    print(f"--- {sname}")
    for r in [r for r in rows2 if r[0] == sname]:
        print(f"{'':39s} {r[1]:3d} {r[2]:6.0f} {r[3]:6.0f} {r[4]:9.0f} {r[5]:9.2f} {r[6]:11d}")

# =====================================================================
# TABLE 3: speed against distance to the nearest AP (free space)
# =====================================================================
edges = np.arange(0, 90, 10)
res = results["free space (alpha 2, no shadowing)"]
dist = res["dist_nearest"].ravel()
header3 = ["distance_to_nearest_AP_m", "users"] + [f"{n}_mean_Mbps" for n in res["rules"]]
rows3 = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (dist >= lo) & (dist < hi)
    rows3.append([f"{lo}-{hi}", int(sel.sum())] + [res["rules"][n]["rate"].ravel()[sel].mean() for n in res["rules"]])
print("\nTABLE 3: average speed against the distance to the nearest AP (free space)")
print(f"{'distance':>10} {'users':>6} " + " ".join(f"{n:>26s}" for n in res["rules"]))
for r in rows3:
    print(f"{r[0]:>8} m {r[1]:6d} " + " ".join(f"{v:23.0f} Mbps" for v in r[2:]))

# =====================================================================
# TABLE 4: noise or interference? (free space and paper-like)
# =====================================================================
header4 = ["propagation", "rule", "real_mean_Mbps", "without_noise_mean_Mbps", "without_interference_mean_Mbps"]
rows4 = []
for sname in SETTINGS:
    for rname, o in results[sname]["rules"].items():
        real = o["rate"].mean()
        no_noise = (config.B_HZ * np.log2(1 + o["signal"] / np.maximum(o["interference"], 1e-30)) / 1e6).mean()
        no_interf = (config.B_HZ * np.log2(1 + o["signal"] / NOISE_W) / 1e6).mean()
        rows4.append([sname, rname, real, no_noise, no_interf])
print("\nTABLE 4: what limits the speed? (average speed in Mbps)")
print(f"{'rule':27s} {'real':>8} {'no noise':>9} {'no interference':>16}")
for sname in SETTINGS:
    print(f"--- {sname}")
    for r in [r for r in rows4 if r[0] == sname]:
        print(f"{r[1]:27s} {r[2]:8.0f} {r[3]:9.0f} {r[4]:16.0f}")


def save(name, header, rows):
    with open(name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[f"{v:.4g}" if isinstance(v, float) else v for v in row] for row in rows])


save("step8a_results.csv", header1, rows1)
save("step8a_L_sweep.csv", header2, rows2)
save("step8a_speed_vs_distance.csv", header3, rows3)
save("step8a_limits.csv", header4, rows4)
print("\nSaved step8a_results.csv, step8a_L_sweep.csv, step8a_speed_vs_distance.csv, step8a_limits.csv")