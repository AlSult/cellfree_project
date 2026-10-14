# The network at the size of the paper: 121 APs on an 11 x 11 grid, 100 users, 1 km x 1 km,
# 64 antennas per AP (Table I).
# Everything is saved as tables in the folder tables/ (next to this script).

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the files of THIS folder always win
import csv
import time
import numpy as np

import paper_config as cfg
import paper_layout as layout
import paper_serving as serving
import paper_simulate as sim
import paper_paths as paths
from paper_rates import NOISE_W


ap_pos = layout.generate_ap_positions()
SETTINGS = {"free space (alpha 2, no shadowing)": cfg.FREE_SPACE,
            "paper-like (alpha 3.8, shadowing 8 dB)": cfg.PAPER_LIKE}


def save(name, header, rows):
    with open(name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[f"{v:.4g}" if isinstance(v, float) else v for v in row] for row in rows])


# =====================================================================
# TABLE 0: the size, and how long one drop takes
# =====================================================================
start = time.perf_counter()
sim.simulate(ap_pos, sim.standard_rules(), n_drops=10, **cfg.PAPER_LIKE)
seconds_per_drop = (time.perf_counter() - start) / 10
print("TABLE 0: THE NETWORK")
print(f"   {cfg.N_AP} APs on a jittered {cfg.GRID_SIDE} x {cfg.GRID_SIDE} grid ({cfg.AP_SPACING_M:.1f} m apart), {cfg.N_UE} users, "
      f"{cfg.AREA_M:.0f} m x {cfg.AREA_M:.0f} m, {cfg.N_ANTENNAS} antennas per AP")
print(f"   users per AP on average: {cfg.N_UE / cfg.N_AP:.2f}   (the paper: 100 / 121 = {100 / 121:.2f})   users per AP served by a cluster: L = {cfg.L_PER_AP}")
print(f"   {cfg.N_DROPS} drops per setting; one drop (three rules, 64 antennas) takes {seconds_per_drop:.2f} s on this computer")
print(f"   channel of one drop: {cfg.N_AP * cfg.N_UE * cfg.N_ANTENNAS:,} complex numbers = {cfg.N_AP * cfg.N_UE * cfg.N_ANTENNAS * 16 / 1e6:.1f} MB\n")

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
L_VALUES = [1, 2, 3, 4, 5, 7, 10]
for sname, s in SETTINGS.items():
    for L in L_VALUES:
        rules = {"c": (lambda beta, L=L: serving.cluster_mask(beta, L))}
        o = sim.simulate(ap_pos, rules, **s)["rules"]["c"]
        st = sim.summarize(o["rate"])
        rows2.append([sname, L, st["mean_Mbps"], st["p5_Mbps"], st["sum_per_drop_Mbps"], o["n_aps"].mean(), o["uncovered"]])
print("\nTABLE 2: the number of users each AP serves (L). L = 7 is the layout of the paper's output vector (121 x 7 + 100 = 947)")
print(f"{'':39s} {'L':>3} {'mean':>6} {'p5':>6} {'sum/drop':>9} {'APs/user':>9} {'not served':>11}")
for sname in SETTINGS:
    print(f"--- {sname}")
    for r in [r for r in rows2 if r[0] == sname]:
        print(f"{'':39s} {r[1]:3d} {r[2]:6.0f} {r[3]:6.0f} {r[4]:9.0f} {r[5]:9.2f} {r[6]:11d}")

# =====================================================================
# TABLE 3: speed against distance to the nearest AP
# =====================================================================
edges = np.arange(0, 80, 10)
header3 = ["propagation", "distance_to_nearest_AP_m", "users"] + [f"{n}_mean_Mbps" for n in sim.standard_rules()]
rows3 = []
for sname in SETTINGS:
    res = results[sname]
    dist = res["dist_nearest"].ravel()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (dist >= lo) & (dist < hi)
        rows3.append([sname, f"{lo}-{hi}", int(sel.sum())] +
                     [res["rules"][n]["rate"].ravel()[sel].mean() if sel.any() else float("nan") for n in res["rules"]])
print("\nTABLE 3: average speed (Mbps) against the distance to the nearest AP")
names = list(sim.standard_rules())
print(f"{'':39s} {'distance':>9} {'users':>6} " + " ".join(f"{n:>26s}" for n in names))
for sname in SETTINGS:
    print(f"--- {sname}")
    for r in [r for r in rows3 if r[0] == sname]:
        print(f"{'':39s} {r[1]:>7} m {r[2]:6d} " + " ".join(f"{v:26.0f}" for v in r[3:]))

# =====================================================================
# TABLE 4: noise or interference?
# =====================================================================
header4 = ["propagation", "rule", "real_mean_Mbps", "without_noise_mean_Mbps", "without_interference_mean_Mbps"]
rows4 = []
for sname in SETTINGS:
    for rname, o in results[sname]["rules"].items():
        real = o["rate"].mean()
        no_noise = (cfg.B_HZ * np.log2(1 + o["signal"] / np.maximum(o["interference"], 1e-30)) / 1e6).mean()
        no_interf = (cfg.B_HZ * np.log2(1 + o["signal"] / NOISE_W) / 1e6).mean()
        rows4.append([sname, rname, real, no_noise, no_interf])
print("\nTABLE 4: what limits the speed? (average speed in Mbps)")
print(f"{'rule':27s} {'real':>8} {'no noise':>9} {'no interference':>16}")
for sname in SETTINGS:
    print(f"--- {sname}")
    for r in [r for r in rows4 if r[0] == sname]:
        print(f"{r[1]:27s} {r[2]:8.0f} {r[3]:9.0f} {r[4]:16.0f}")

# =====================================================================
# TABLE 5: the middle network against the paper-size network
# =====================================================================
header5 = ["propagation", "rule", "middle_mean_Mbps", "paper_size_mean_Mbps", "middle_p5_Mbps", "paper_size_p5_Mbps"]
rows5 = []
middle_path = paths.find_middle_results()
if middle_path:
    with open(middle_path, encoding="utf-8") as f:
        middle = {(r["propagation"], r["rule"].replace("L=3", f"L={cfg.L_PER_AP}")): r for r in csv.DictReader(f)}
    for r in rows1:
        m = middle.get((r[0], r[1]))
        if m:
            rows5.append([r[0], r[1], float(m["mean_Mbps"]), r[2], float(m["p5_Mbps"]), r[4]])
    print("\nTABLE 5: the middle network (25 APs, 20 users, 16 antennas) against the paper size (121 APs, 100 users, 64 antennas)")
    print(f"{'rule':27s} {'mean: middle':>13} {'paper size':>11} | {'p5: middle':>11} {'paper size':>11}")
    for sname in SETTINGS:
        print(f"--- {sname}")
        for r in [r for r in rows5 if r[0] == sname]:
            print(f"{r[1]:27s} {r[2]:13.0f} {r[3]:11.0f} | {r[4]:11.0f} {r[5]:11.0f}")
else:
    print("\nTABLE 5 skipped: step8a_results.csv not found in cellfree_project (run step8a_run.py there first)")

# =====================================================================
# TABLE 6: 16 or 64 antennas?
# =====================================================================
header6 = ["antennas", "rule", "mean_Mbps", "p5_Mbps"]
rows6 = []
for n_ant in (16, 64):
    res = sim.simulate(ap_pos, sim.standard_rules(), n_ant=n_ant, **cfg.PAPER_LIKE)
    for rname, o in res["rules"].items():
        st = sim.summarize(o["rate"])
        rows6.append([n_ant, rname, st["mean_Mbps"], st["p5_Mbps"]])
print("\nTABLE 6: antennas per AP, paper-like propagation, paper-size network")
print(f"{'rule':27s} {'mean: 16 ant.':>14} {'64 ant.':>9} | {'p5: 16 ant.':>12} {'64 ant.':>9}")
for rname in sim.standard_rules():
    a = [r for r in rows6 if r[1] == rname and r[0] == 16][0]
    b = [r for r in rows6 if r[1] == rname and r[0] == 64][0]
    print(f"{rname:27s} {a[2]:14.0f} {b[2]:9.0f} | {a[3]:12.0f} {b[3]:9.0f}")

save(paths.table("results.csv"), header1, rows1)
save(paths.table("L_sweep.csv"), header2, rows2)
save(paths.table("speed_vs_distance.csv"), header3, rows3)
save(paths.table("limits.csv"), header4, rows4)
if rows5:
    save(paths.table("vs_middle.csv"), header5, rows5)
save(paths.table("antennas.csv"), header6, rows6)
print("\nSaved in the folder tables/: results.csv, L_sweep.csv, speed_vs_distance.csv, limits.csv, "
      "vs_middle.csv, antennas.csv")