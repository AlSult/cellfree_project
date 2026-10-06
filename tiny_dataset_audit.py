# The audit of the tiny dataset:
# it reads ONLY the delivered files (tiny_dataset.npz and tiny_dataset_info.json) and
# re-computes everything with its own, independent formulas (numpy only).
# The only exception is it runs the generator again to test reproducibility.
#
# It prints the checks, and SAVES (in the folder audit_output/):
#   audit_results.csv      the table of all checks (open it in Excel)
#   audit_results.md       the same table, ready to paste into a report
#   audit_deviations.csv   how the tiny network differs from Table I of the paper
#   audit_summary.png      checks passed per group + how close each check is to its limit
#   audit_evidence.png     nine pictures that show the evidence behind the checks

import csv
import json
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

d = dict(np.load("tiny_dataset/tiny_dataset.npz"))
info = json.load(open("tiny_dataset/tiny_dataset_info.json"))
E, T, M, K, N = info["episodes"], info["steps"], info["n_ap"], info["n_ue"], info["n_antennas"]
RULES = info["rules"]
B, FC, C = info["b_hz"], info["fc_hz"], info["c"]
NOISE_W = 10 ** ((info["n0_dbm_hz"] + 10 * np.log10(B) + info["nf_db"] - 30) / 10)
BUDGET = 10 ** ((info["ptx_dbm"] - 30) / 10)
DT = info["dt_s"]
OUT = "audit_output"
os.makedirs(OUT, exist_ok=True)

GROUPS = {"A": "Structure", "B": "Plausibility", "C": "Independent recomputation", "D": "Consistency",
          "E": "Random part", "F": "Stress tests", "G": "Reproducibility"}
PANEL = {"B3": 1, "C2": 1, "B4": 2, "C5": 3, "D1": 4, "D2": 4, "D3": 4, "E2": 5, "E3": 6, "E4": 6, "E5": 6,
         "E6": 7, "E7": 7, "F1": 8, "D5": 9}          # which picture of audit_evidence.png shows the check
records = []


def check(code, text, ok, detail="", value=None, limit=None):
    """Prints one check and remembers it for the table.
    value / limit: the measured number and the largest value that still passes (if the check is about a number)."""
    ok = bool(ok)
    records.append({"code": code, "group": GROUPS[code[0]], "check": text, "result": "PASS" if ok else "FAIL",
                    "measured": value, "limit": limit,
                    "margin": None if value is None else max(abs(value), 1e-16) / limit,
                    "panel": PANEL.get(code, ""), "detail": detail})
    print(f"[{code}] {text}{' (' + detail + ')' if detail else ''} -> {'PASS' if ok else 'FAIL'}")


def note(text):
    print(f"      note: {text}")


def my_rates(H, p):
    """Own implementation of the whole chain, with explicit loops.
    H: (M, K, N) channel, p: (M, K) power of every beam. Returns signal, interference, sinr, rate."""
    W = np.zeros_like(H)
    for m in range(M):
        for j in range(K):
            W[m, j] = np.sqrt(p[m, j]) * H[m, j] / np.linalg.norm(H[m, j])     # aim the beam at the channel
    sig, itf = np.zeros(K), np.zeros(K)
    for k in range(K):
        for j in range(K):
            amp = sum(np.vdot(H[m, k], W[m, j]) for m in range(M))             # all APs add up (coherent)
            if j == k:
                sig[k] = abs(amp) ** 2
            else:
                itf[k] += abs(amp) ** 2
    sinr = sig / (itf + NOISE_W)
    return sig, itf, sinr, B * np.log2(1 + sinr) / 1e6


print("A. STRUCTURE: is the file complete and tidy?")
needed = ["ap_pos", "pos", "speed", "dist", "beta", "H"] + [f"{n}_{r}" for r in RULES for n in
          ("mask", "power", "signal", "interference", "sinr", "rate")]
check("A1", "all expected arrays are in the file", all(n in d for n in needed), f"{len(needed)} arrays")
shapes = {"pos": (E, T, K, 2), "speed": (E, K), "dist": (E, T, M, K), "beta": (E, T, M, K), "H": (E, T, M, K, N)}
for r in RULES:
    shapes.update({f"mask_{r}": (E, T, M, K), f"power_{r}": (E, T, M, K)})
    shapes.update({f"{n}_{r}": (E, T, K) for n in ("signal", "interference", "sinr", "rate")})
check("A2", "every array has the shape the documentation promises", all(d[n].shape == s for n, s in shapes.items()))
check("A3", "no NaN or infinite value anywhere", all(np.all(np.isfinite(d[n])) for n in d if d[n].dtype != bool))
check("A4", "data types are right (H complex, masks True/False, the rest real numbers)",
      d["H"].dtype == complex and all(d[f"mask_{r}"].dtype == bool for r in RULES) and d["beta"].dtype == float)
check("A5", "the info file documents units and the random seeds",
      all(key in info for key in ("units", "walk_seed0", "channel_seed0", "array_axes")))
distinct = all(not np.array_equal(d["pos"][i], d["pos"][j]) and not np.array_equal(d["H"][i], d["H"][j])
               for i in range(E) for j in range(i + 1, E))
check("A6", "no two episodes are copies of each other", distinct)

print("\nB. PLAUSIBILITY: are the values in a believable range?")
check("B1", "everybody is inside the map", np.all((d["pos"] >= 0) & (d["pos"] <= info["area_m"])))
check("B2", "distances are positive and not larger than the diagonal of the map",
      np.all(d["dist"] > 0) and np.all(d["dist"] <= info["area_m"] * np.sqrt(2) + 1e-9))
pl = -10 * np.log10(d["beta"])
check("B3", "path loss is between 40 and 100 dB", np.all((pl > 40) & (pl < 100)), f"{pl.min():.1f} to {pl.max():.1f} dB")
ok_b4 = True
bound_all, stored_all, rule_all = [], [], []
for r in RULES:
    bound = B * np.log2(1 + d[f"signal_{r}"] / NOISE_W) / 1e6          # speed if nobody else existed
    ok_b4 &= bool(np.all(d[f"sinr_{r}"] >= 0) and np.all(d[f"rate_{r}"] >= 0) and np.all(d[f"rate_{r}"] <= bound + 1e-9))
    bound_all.append(bound.ravel())
    stored_all.append(d[f"rate_{r}"].ravel())
    rule_all.append(np.full(bound.size, r))
check("B4", "speeds are never negative and never above the 'no interference' limit", ok_b4)
ok_b5 = all(np.all(d[f"power_{r}"] >= 0) and np.all(d[f"power_{r}"].sum(axis=3) <= BUDGET + 1e-9) for r in RULES)
check("B5", "no AP ever uses more than its power budget", ok_b5, f"budget {BUDGET:.1f} W")
check("B6", "noise floor is -85 dBm (-174 + 80 + 9)", abs(info["noise_dbm"] - (-85)) < 1e-9)
rates_all = np.concatenate([d[f"rate_{r}"].ravel() for r in RULES])
note(f"speeds in the file go from {rates_all.min():.0f} to {rates_all.max():.0f} Mbps (100 MHz of bandwidth)")

print("\nC. INDEPENDENT RECOMPUTATION: do my own formulas give the same numbers?")
my_dist = np.hypot(d["ap_pos"][None, None, :, None, 0] - d["pos"][:, :, None, :, 0],
                   d["ap_pos"][None, None, :, None, 1] - d["pos"][:, :, None, :, 1])
err_dist = np.max(np.abs(my_dist - d["dist"]))
check("C1", "distances recomputed from the positions", np.allclose(my_dist, d["dist"]),
      f"largest difference {err_dist:.1e} m", err_dist, 1e-8)
my_pl = 32.44 + 20 * np.log10(FC / 1e6) + 20 * np.log10(d["dist"] / 1000)      # the engineers' rule of thumb
err_pl = np.max(np.abs(my_pl - pl))
check("C2", "path loss recomputed with the rule of thumb 32.44 + 20log10(f/MHz) + 20log10(d/km)",
      err_pl < 0.01, f"largest difference {err_pl:.4f} dB", err_pl, 0.01)
nearest = np.zeros((E, T, M, K), dtype=bool)
np.put_along_axis(nearest, np.argmin(d["dist"], axis=2)[:, :, None, :], True, axis=2)
check("C3", "the nearest-AP rule: every user has exactly one AP and it is the closest one",
      np.array_equal(nearest, d["mask_nearest"]) and np.all(d["mask_nearest"].sum(axis=2) == 1)
      and np.all(d["mask_cellfree"]))
for r in RULES:
    expected_p = np.where(d[f"mask_{r}"], BUDGET / K, 0.0)
    check(f"C4{r[0]}", f"power of every beam is budget / {K} where a beam exists ({r} rule)", np.allclose(d[f"power_{r}"], expected_p))
worst = 0.0
mine_rate_all, stored_rate_all = [], []
for r in RULES:
    for e in range(E):
        for t in range(T):
            s, i, q, rate = my_rates(d["H"][e, t], d[f"power_{r}"][e, t])
            for mine, stored in ((s, d[f"signal_{r}"][e, t]), (i, d[f"interference_{r}"][e, t]),
                                 (q, d[f"sinr_{r}"][e, t]), (rate, d[f"rate_{r}"][e, t])):
                worst = max(worst, np.max(np.abs(mine - stored) / np.abs(stored)))
            mine_rate_all.append(rate)
            stored_rate_all.append(d[f"rate_{r}"][e, t])
mine_rate_all, stored_rate_all = np.concatenate(mine_rate_all), np.concatenate(stored_rate_all)
check("C5", f"signal, interference, SINR and speed recomputed from H for all {2*E*T} snapshots",
      worst < 1e-8, f"largest relative difference {worst:.1e}", worst, 1e-8)
err_rate = max(np.max(np.abs(d[f"rate_{r}"] - B * np.log2(1 + d[f"sinr_{r}"]) / 1e6) / d[f"rate_{r}"]) for r in RULES)
check("C6", "speed = bandwidth x log2(1 + SINR)", err_rate < 1e-5, f"largest relative difference {err_rate:.1e}", err_rate, 1e-5)

print("\nD. CONSISTENCY: do the pieces agree with each other?")
step = np.linalg.norm(np.diff(d["pos"], axis=1), axis=3) / DT
ratio = step / d["speed"][:, None, :]
check("D1", "nobody moves faster than its own speed in any second", np.all(ratio <= 1 + 1e-9),
      f"largest step / speed = {ratio.max():.12f}", max(ratio.max() - 1, 0), 1e-9)
exact = np.abs(ratio - 1) < 1e-6
check("D2", "almost all steps are exactly speed x time (the others are bounces)", exact.mean() > 0.9,
      f"{exact.mean()*100:.1f}% exact", 1 - exact.mean(), 0.1)
excess = np.max(np.abs(np.diff(d["dist"], axis=1)) - step[:, :, None, :] * DT)
check("D3", "the distance to an AP can never change by more than the user walked", excess <= 1e-9,
      f"largest excess {max(excess, 0):.1e} m", max(excess, 0), 1e-9)
check("D4", "all episodes start where the documentation says",
      np.allclose(d["pos"][:, 0], np.array(info["ue_start"])[None]))
short = np.max((d["signal_nearest"] - d["signal_cellfree"]) / d["signal_nearest"])
check("D5", "the cell-free wanted signal is never weaker than the nearest-AP one (same channel)",
      short <= 1e-12, f"largest shortfall {max(short, 0):.1e}", max(short, 0), 1e-12)
check("D6", "cell-free: every AP uses exactly its whole budget",
      np.allclose(d["power_cellfree"].sum(axis=3), BUDGET))
ties = np.sum(np.abs(d["dist"][:, :, 0, :] - d["dist"][:, :, 1, :]) < 1e-9)
note(f"{ties} snapshots have a user exactly equally far from both APs (a tie, resolved in favour of AP0)")

print("\nE. THE RANDOM PART: does the flicker behave like Rayleigh fading?")
g = d["H"] / np.sqrt(d["beta"])[..., None]            # remove the path loss: what is left is the random part
n = g.size
power = np.abs(g) ** 2
e1 = max(abs(g.mean()) / 0.025, abs(power.mean() - 1) / 0.02, abs(g.real.var() - 0.5) / 0.015, abs(g.imag.var() - 0.5) / 0.015)
check("E1", "random part: average 0, average power 1, real and imaginary variance 0.5", e1 < 1,
      f"mean {abs(g.mean()):.4f}, power {power.mean():.3f}, var {g.real.var():.3f}/{g.imag.var():.3f}", e1, 1.0)
x = np.sort(power.ravel())
ks = np.max(np.abs(np.arange(1, n + 1) / n - (1 - np.exp(-x))))
check("E2", "antenna powers follow the exponential law of Rayleigh fading (Kolmogorov-Smirnov test)",
      ks < 1.63 / np.sqrt(n), f"distance {ks:.4f}, limit {1.63/np.sqrt(n):.4f}", ks, 1.63 / np.sqrt(n))
corr_ant = np.corrcoef(g[..., 0].real.ravel(), g[..., 1].real.ravel())[0, 1]
check("E3", "different antennas are independent", abs(corr_ant) < 0.08, f"correlation {corr_ant:+.3f}", abs(corr_ant), 0.08)
corr_time = np.corrcoef(g[:, :-1].real.ravel(), g[:, 1:].real.ravel())[0, 1]
check("E4", "one second and the next one are independent (a new draw every second, as documented)",
      abs(corr_time) < 0.025, f"correlation {corr_time:+.3f}", abs(corr_time), 0.025)
corr_pair = np.corrcoef(g[:, :, 0, 0, :].real.ravel(), g[:, :, 1, 2, :].real.ravel())[0, 1]
check("E5", "different AP-user pairs are independent", abs(corr_pair) < 0.06, f"correlation {corr_pair:+.3f}", abs(corr_pair), 0.06)
strength = np.sum(power, axis=4)
check("E6", f"average channel strength is {N} x beta (array gain)", abs(strength.mean() / N - 1) < 0.02,
      f"{strength.mean()/N:.3f}", abs(strength.mean() / N - 1), 0.02)
rel = strength.std() / strength.mean()
check("E7", f"channel hardening: relative randomness of the strength is 1/sqrt({N}) = {1/np.sqrt(N):.3f}",
      abs(rel / (1 / np.sqrt(N)) - 1) < 0.05, f"measured {rel:.3f}", abs(rel / (1 / np.sqrt(N)) - 1), 0.05)

print("\nF. STRESS TESTS: does the answer change when it must not (and only then)?")
sample = [(e, t) for e in range(0, E, 3) for t in range(0, T, 8)]
half_rates, full_rates = [], []
for e, t in sample:
    half_rates.append(my_rates(d["H"][e, t], d["power_cellfree"][e, t] / 2)[3])
    full_rates.append(d["rate_cellfree"][e, t])
half_rates, full_rates = np.concatenate(half_rates), np.concatenate(full_rates)
p_half = np.max(np.abs(half_rates - full_rates) / full_rates)
check("F1", "halving every power changes the speeds by less than 1% (noise is negligible)", p_half < 0.01,
      f"{p_half*100:.4f}%", p_half, 0.01)
perm = np.arange(K)[::-1]
ok_f2 = all(np.allclose(my_rates(d["H"][e, t][:, perm], d["power_cellfree"][e, t][:, perm])[3],
                        d["rate_cellfree"][e, t][perm]) for e, t in sample)
check("F2", "renumbering the users just renumbers the results", ok_f2)
ok_f3 = all(np.allclose(my_rates(d["H"][e, t][::-1], d["power_nearest"][e, t][::-1])[3], d["rate_nearest"][e, t]) for e, t in sample)
check("F3", "renumbering the APs changes nothing", ok_f3)
ok_f4 = True
for e, t in sample:
    p0 = d["power_cellfree"][e, t].copy()
    p0[:, 1] = 0.0                                            # switch off every beam meant for user 1
    s, i, q, rate = my_rates(d["H"][e, t], p0)
    ok_f4 &= bool(s[1] == 0 and rate[1] == 0 and np.all(i[[0, 2, 3]] < d["interference_cellfree"][e, t][[0, 2, 3]]))
check("F4", "switching off the beams of one user: its speed is 0 and everybody else hears less interference", ok_f4)

print("\nG. REPRODUCIBILITY: does running the generator again give the same file?")
try:
    from tiny_dataset_export import build_dataset
    again, _ = build_dataset(E, info["walk_seed0"], info["channel_seed0"])
    other, _ = build_dataset(1, info["walk_seed0"] + 1, info["channel_seed0"] + 1)
    check("G1", "same seeds -> every array is identical to the delivered file", all(np.array_equal(again[n], d[n]) for n in d))
    check("G2", "other seeds -> different data", not np.array_equal(other["H"][0], d["H"][0]))
except ImportError as error:
    note(f"G skipped: {error}")

print("\nH. DOCUMENTATION: how does this tiny network differ from Table I of the paper? (information, not pass/fail)")
paper = {"antennas per AP": (N, 64), "area": (f"{info['area_m']:.0f} m x {info['area_m']:.0f} m", "1 km x 1 km"),
         "APs": (M, "121 (20 federated)"), "users": (K, 100),
         "path loss exponent": (2.0, 3.8), "shadow fading": ("none", "8 dB"),
         "channel model": ("free space + Rayleigh", "3GPP TR 38.901"),
         "mobility": ("straight walk, bounces", "Random Waypoint"),
         "speeds": (f"{info['speed_min_ms']*3.6:.0f} to {info['speed_max_ms']*3.6:.0f} km/h", "3 to 60 km/h")}
same = {"carrier": (info["fc_hz"] / 1e9, 3.5), "bandwidth (MHz)": (B / 1e6, 100), "power (dBm)": (info["ptx_dbm"], 46),
        "noise figure (dB)": (info["nf_db"], 9), "noise density (dBm/Hz)": (info["n0_dbm_hz"], -174)}
deviations = []
print(f"      {'item':24s} {'ours':>26s} {'paper':>20s}")
for name, (ours, theirs) in {**same, **paper}.items():
    verdict = "same" if str(ours) == str(theirs) or ours == theirs else "DIFFERENT"
    deviations.append([name, ours, theirs, verdict])
    print(f"      {name:24s} {str(ours):>26s} {str(theirs):>20s}   {verdict}")
same_start = bool(np.all(d["pos"][:, 0] == d["pos"][0, 0]))
deviations.append(["all episodes start from the same positions", same_start, "100 runs with different topologies", "NOTE"])
print(f"      all episodes start from the same 4 positions: {same_start}")

passed = sum(r["result"] == "PASS" for r in records)
print(f"\nAUDIT RESULT: {passed} of {len(records)} checks passed" + ("" if passed == len(records) else
      "  -> FAILED: " + ", ".join(r["code"] for r in records if r["result"] == "FAIL")))


# =====================================================================
# SAVE THE TABLES
# =====================================================================
def fmt(v):
    return "" if v is None else f"{v:.3g}"


header = ["code", "group", "check", "result", "measured", "limit", "margin", "picture", "detail"]
rows = [[r["code"], r["group"], r["check"], r["result"], fmt(r["measured"]), fmt(r["limit"]), fmt(r["margin"]),
         f"panel {r['panel']}" if r["panel"] else "", r["detail"]] for r in records]
with open(f"{OUT}/audit_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
with open(f"{OUT}/audit_deviations.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item", "tiny network", "paper (Table I)", "verdict"])
    w.writerows(deviations)
with open(f"{OUT}/audit_results.md", "w", encoding="utf-8") as f:
    f.write(f"# Audit of the tiny dataset: {passed} of {len(records)} checks passed\n\n")
    f.write("margin = measured / limit. Below 1 passes; the smaller, the safer.\n\n")
    f.write("| code | group | check | result | measured | limit | margin | picture |\n|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(str(c).replace("|", "/") for c in r[:8]) + " |\n")
    f.write("\n## Differences from Table I of the paper\n\n| item | tiny network | paper | verdict |\n|---|---|---|---|\n")
    for dev in deviations:
        f.write("| " + " | ".join(str(c) for c in dev) + " |\n")

# =====================================================================
# PICTURE 1: summary
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.7]})
letters = list(GROUPS)
n_pass = [sum(r["code"][0] == L and r["result"] == "PASS" for r in records) for L in letters]
n_fail = [sum(r["code"][0] == L and r["result"] == "FAIL" for r in records) for L in letters]
ypos = np.arange(len(letters))[::-1]
ax[0].barh(ypos, n_pass, color="tab:green", label="passed")
ax[0].barh(ypos, n_fail, left=n_pass, color="tab:red", label="FAILED")
for yy, p_, f_ in zip(ypos, n_pass, n_fail):
    ax[0].text(p_ + f_ + 0.15, yy, f"{p_} of {p_ + f_}", va="center")
ax[0].set_yticks(ypos)
ax[0].set_yticklabels([f"{L}. {GROUPS[L]}" for L in letters])
ax[0].set_xlabel("number of checks")
ax[0].set_xlim(0, max(np.array(n_pass) + np.array(n_fail)) + 2)
ax[0].set_title(f"Checks per group: {passed} of {len(records)} passed")
ax[0].legend(loc="lower right")

numeric = sorted([r for r in records if r["margin"] is not None], key=lambda r: r["margin"])
ypos2 = np.arange(len(numeric))
colours = ["tab:green" if r["margin"] < 0.8 else "tab:orange" if r["margin"] < 1 else "tab:red" for r in numeric]
ax[1].hlines(ypos2, 1e-17, [max(r["margin"], 1e-16) for r in numeric], color="lightgray", lw=1)
ax[1].scatter([max(r["margin"], 1e-16) for r in numeric], ypos2, s=90, color=colours, zorder=3)
ax[1].set_xscale("log")
ax[1].set_xlim(1e-17, 30)
ax[1].axvline(1, color="red", lw=2)
ax[1].text(0.7, -0.4, "limit: a check fails\nbeyond this line", color="red", fontsize=9, va="bottom", ha="right")
for yy, r in zip(ypos2, numeric):
    ax[1].text(max(r["margin"], 1e-16) * 1.8, yy, f"{r['margin']:.2g}", va="center", fontsize=8)
ax[1].set_yticks(ypos2)
ax[1].set_yticklabels([f"{r['code']}  {r['check'][:52]}" for r in numeric], fontsize=8)
ax[1].set_xlabel("margin = measured value / limit   (log scale, smaller = safer)")
ax[1].set_title("How close is each numeric check to its limit?\n(a dot near the red line would be a warning sign)")
ax[1].grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/audit_summary.png", dpi=130)

# =====================================================================
# PICTURE 2: the evidence
# =====================================================================
rng_plot = np.random.default_rng(0)                      # only used to thin out points for drawing
fig, ax = plt.subplots(3, 3, figsize=(18, 15))

# 1: path loss against distance (B3, C2)
dist_flat, pl_flat = d["dist"].ravel(), pl.ravel()
pick = rng_plot.choice(dist_flat.size, 3000, replace=False)
ax[0, 0].scatter(dist_flat[pick], pl_flat[pick], s=7, alpha=0.5, label="data in the file", zorder=3)
dd = np.logspace(np.log10(dist_flat.min() * 0.9), np.log10(dist_flat.max() * 1.1), 200)
ax[0, 0].plot(dd, 32.44 + 20 * np.log10(FC / 1e6) + 20 * np.log10(dd / 1000), "k-", lw=5, alpha=0.5, label="rule of thumb (my own formula)")
ax[0, 0].axhline(40, color="red", ls="--")
ax[0, 0].axhline(100, color="red", ls="--", label="limits of check B3")
ax[0, 0].set_xscale("log")
ax[0, 0].set_xlabel("distance (m)")
ax[0, 0].set_ylabel("path loss (dB)")
ax[0, 0].set_title("1. B3, C2: path loss follows the independent formula\nand stays inside the limits")
ax[0, 0].legend(fontsize=8)
ax[0, 0].grid(True, which="both", alpha=0.3)

# 2: speed against the no-interference limit (B4)
bound_c, stored_c, rule_c = np.concatenate(bound_all), np.concatenate(stored_all), np.concatenate(rule_all)
for r, colour in zip(RULES, ["tab:gray", "tab:blue"]):
    sel = np.where(rule_c == r)[0]
    sel = rng_plot.choice(sel, 1500, replace=False)
    ax[0, 1].scatter(bound_c[sel], stored_c[sel], s=7, alpha=0.4, color=colour, label=f"{r} rule")
top = bound_c.max() * 1.05
ax[0, 1].plot([0, top], [0, top], "r--", label="speed = limit (nobody else existed)")
ax[0, 1].set_xlabel("speed if nobody else existed (Mbps)")
ax[0, 1].set_ylabel("speed in the file (Mbps)")
ax[0, 1].set_title("2. B4: every speed is below the no-interference limit")
ax[0, 1].legend(fontsize=8)
ax[0, 1].grid(True, alpha=0.3)

# 3: stored against recomputed (C5)
pick = rng_plot.choice(stored_rate_all.size, 2000, replace=False)
ax[0, 2].scatter(stored_rate_all[pick], mine_rate_all[pick], s=7, alpha=0.4)
top = stored_rate_all.max() * 1.05
ax[0, 2].plot([0, top], [0, top], "r--", label="identical")
ax[0, 2].set_xlabel("speed stored in the file (Mbps)")
ax[0, 2].set_ylabel("speed recomputed by the audit (Mbps)")
ax[0, 2].set_title(f"3. C5: the audit's own calculation gives the same speeds\n(largest relative difference {worst:.1e})")
ax[0, 2].legend()
ax[0, 2].grid(True, alpha=0.3)

# 4: step / speed (D1, D2, D3)
ax[1, 0].hist(ratio.ravel(), bins=np.linspace(0, 1.05, 106), color="tab:blue")
ax[1, 0].set_yscale("log")
ax[1, 0].axvline(1, color="red", ls="--", label="a step can never be longer than speed x time")
ax[1, 0].set_xlabel("length of a step / (speed x time)")
ax[1, 0].set_ylabel("number of steps (log scale)")
ax[1, 0].set_title(f"4. D1-D3: {exact.mean()*100:.1f}% of the steps are exactly 1.0,\nthe few shorter ones are bounces off a wall")
ax[1, 0].legend(fontsize=8, loc="upper center")

# 5: Q-Q plot against the exponential law (E2)
probs = np.linspace(0.005, 0.995, 200)
ax[1, 1].plot(-np.log(1 - probs), np.quantile(power.ravel(), probs), "o", ms=4, label="antenna powers in the file")
ax[1, 1].plot([0, 5.5], [0, 5.5], "r--", label="exactly the exponential law")
ax[1, 1].set_xlabel("expected value for a Rayleigh channel")
ax[1, 1].set_ylabel("value in the file")
ax[1, 1].set_title(f"5. E2: the random antenna powers follow the exponential law\n(Kolmogorov-Smirnov distance {ks:.4f}, limit {1.63/np.sqrt(n):.4f})")
ax[1, 1].legend()
ax[1, 1].grid(True, alpha=0.3)

# 6: independence (E3, E4, E5)
names = ["antennas\n(E3)", "seconds\n(E4)", "AP-user pairs\n(E5)"]
values, limits = [abs(corr_ant), abs(corr_time), abs(corr_pair)], [0.08, 0.025, 0.06]
ax[1, 2].bar(range(3), values, color="tab:blue", label="measured |correlation|")
ax[1, 2].scatter(range(3), limits, marker="_", s=2200, color="red", linewidths=3, label="limit", zorder=3)
for i, v in enumerate(values):
    ax[1, 2].text(i, v + 0.002, f"{v:.3f}", ha="center")
ax[1, 2].set_xticks(range(3))
ax[1, 2].set_xticklabels(names)
ax[1, 2].set_ylim(0, 0.1)
ax[1, 2].set_ylabel("absolute correlation (0 = independent)")
ax[1, 2].set_title("6. E3-E5: the random numbers are independent")
ax[1, 2].legend()

# 7: channel strength (E6, E7)
xs = np.linspace(0, 40, 400)
ax[2, 0].hist(strength.ravel(), bins=60, range=(0, 40), density=True, alpha=0.6, label="channel strength in the file")
ax[2, 0].plot(xs, xs ** (N - 1) * np.exp(-xs) / math.gamma(N), "k-", lw=2, label=f"theory (sum of {N} Rayleigh powers)")
ax[2, 0].axvline(N, color="red", ls="--", label=f"average = {N}")
ax[2, 0].set_xlabel("channel strength / beta")
ax[2, 0].set_ylabel("how often")
ax[2, 0].set_title(f"7. E6, E7: array gain and channel hardening\n(average {strength.mean():.2f}, expected {N})")
ax[2, 0].legend(fontsize=8)

# 8: half power (F1)
ax[2, 1].scatter(full_rates, half_rates, s=8, alpha=0.5)
top = full_rates.max() * 1.05
ax[2, 1].plot([0, top], [0, top], "r--", label="identical")
ax[2, 1].set_xlabel("speed with the normal powers (Mbps)")
ax[2, 1].set_ylabel("speed with every power halved (Mbps)")
ax[2, 1].set_title(f"8. F1: halving all powers changes nothing\n(largest change {p_half*100:.4f}%)")
ax[2, 1].legend()
ax[2, 1].grid(True, alpha=0.3)

# 9: cell-free against nearest signal (D5)
ax[2, 2].scatter(d["signal_nearest"].ravel(), d["signal_cellfree"].ravel(), s=6, alpha=0.4)
lo, hi = d["signal_nearest"].min() * 0.8, d["signal_cellfree"].max() * 1.2
ax[2, 2].plot([lo, hi], [lo, hi], "r--", label="equal")
ax[2, 2].set_xscale("log")
ax[2, 2].set_yscale("log")
ax[2, 2].set_xlabel("wanted signal, nearest-AP rule (W)")
ax[2, 2].set_ylabel("wanted signal, cell-free rule (W)")
ax[2, 2].set_title("9. D5: with the same channel, cell-free is never below\nthe nearest-AP signal (all points on or above the line)")
ax[2, 2].legend()
ax[2, 2].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/audit_evidence.png", dpi=110)

print(f"\nSaved in the folder {OUT}/ : audit_results.csv, audit_results.md, audit_deviations.csv, "
      f"audit_summary.png, audit_evidence.png")
if "--no-show" not in sys.argv:
    plt.show()