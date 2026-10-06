# The audit of the tiny dataset:
# it reads ONLY the delivered files (tiny_dataset.npz and tiny_dataset_info.json) and
# re-computes everything with its own, independent formulas (numpy only).
# The only exception is it runs the generator again to test reproducibility.


import json
import numpy as np

d = dict(np.load("tiny_dataset.npz"))
info = json.load(open("tiny_dataset_info.json"))
E, T, M, K, N = info["episodes"], info["steps"], info["n_ap"], info["n_ue"], info["n_antennas"]
RULES = info["rules"]
B, FC, C = info["b_hz"], info["fc_hz"], info["c"]
NOISE_W = 10 ** ((info["n0_dbm_hz"] + 10 * np.log10(B) + info["nf_db"] - 30) / 10)
BUDGET = 10 ** ((info["ptx_dbm"] - 30) / 10)
DT = info["dt_s"]
results = []


def check(code, text, ok, detail=""):
    ok = bool(ok)
    results.append((code, ok))
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
for r in RULES:
    bound = B * np.log2(1 + d[f"signal_{r}"] / NOISE_W) / 1e6          # speed if nobody else existed
    ok_b4 &= bool(np.all(d[f"sinr_{r}"] >= 0) and np.all(d[f"rate_{r}"] >= 0) and np.all(d[f"rate_{r}"] <= bound + 1e-9))
check("B4", "speeds are never negative and never above the 'no interference' limit", ok_b4)
ok_b5 = all(np.all(d[f"power_{r}"] >= 0) and np.all(d[f"power_{r}"].sum(axis=3) <= BUDGET + 1e-9) for r in RULES)
check("B5", "no AP ever uses more than its power budget", ok_b5, f"budget {BUDGET:.1f} W")
check("B6", "noise floor is -85 dBm (-174 + 80 + 9)", abs(info["noise_dbm"] - (-85)) < 1e-9)
rates_all = np.concatenate([d[f"rate_{r}"].ravel() for r in RULES])
note(f"speeds in the file go from {rates_all.min():.0f} to {rates_all.max():.0f} Mbps (100 MHz of bandwidth)")

print("\nC. INDEPENDENT RECOMPUTATION: do my own formulas give the same numbers?")
my_dist = np.hypot(d["ap_pos"][None, None, :, None, 0] - d["pos"][:, :, None, :, 0],
                   d["ap_pos"][None, None, :, None, 1] - d["pos"][:, :, None, :, 1])
check("C1", "distances recomputed from the positions", np.allclose(my_dist, d["dist"]))
my_pl = 32.44 + 20 * np.log10(FC / 1e6) + 20 * np.log10(d["dist"] / 1000)      # the engineers' rule of thumb
check("C2", "path loss recomputed with the rule of thumb 32.44 + 20log10(f/MHz) + 20log10(d/km)",
      np.max(np.abs(my_pl - pl)) < 0.01, f"largest difference {np.max(np.abs(my_pl - pl)):.4f} dB")
nearest = np.zeros((E, T, M, K), dtype=bool)
np.put_along_axis(nearest, np.argmin(d["dist"], axis=2)[:, :, None, :], True, axis=2)
check("C3", "the nearest-AP rule: every user has exactly one AP and it is the closest one",
      np.array_equal(nearest, d["mask_nearest"]) and np.all(d["mask_nearest"].sum(axis=2) == 1)
      and np.all(d["mask_cellfree"]))
for r in RULES:
    expected_p = np.where(d[f"mask_{r}"], BUDGET / K, 0.0)
    check(f"C4{r[0]}", f"power of every beam is budget / {K} where a beam exists ({r} rule)", np.allclose(d[f"power_{r}"], expected_p))
worst = 0.0
for r in RULES:
    for e in range(E):
        for t in range(T):
            s, i, q, rate = my_rates(d["H"][e, t], d[f"power_{r}"][e, t])
            for mine, stored in ((s, d[f"signal_{r}"][e, t]), (i, d[f"interference_{r}"][e, t]),
                                 (q, d[f"sinr_{r}"][e, t]), (rate, d[f"rate_{r}"][e, t])):
                worst = max(worst, np.max(np.abs(mine - stored) / np.abs(stored)))
check("C5", f"signal, interference, SINR and speed recomputed from H for all {2*E*T} snapshots",
      worst < 1e-8, f"largest relative difference {worst:.1e}")
check("C6", "speed = bandwidth x log2(1 + SINR)",
      all(np.allclose(d[f"rate_{r}"], B * np.log2(1 + d[f"sinr_{r}"]) / 1e6) for r in RULES))

print("\nD. CONSISTENCY: do the pieces agree with each other?")
step = np.linalg.norm(np.diff(d["pos"], axis=1), axis=3) / DT
check("D1", "nobody moves faster than its own speed in any second", np.all(step <= d["speed"][:, None, :] * (1 + 1e-9)))
exact = np.abs(step - d["speed"][:, None, :]) / d["speed"][:, None, :] < 1e-6
check("D2", "almost all steps are exactly speed x time (the others are bounces)", exact.mean() > 0.9,
      f"{exact.mean()*100:.1f}% exact")
check("D3", "the distance to an AP can never change by more than the user walked",
      np.all(np.abs(np.diff(d["dist"], axis=1)) <= step[:, :, None, :] * DT + 1e-9))
check("D4", "all episodes start where the documentation says",
      np.allclose(d["pos"][:, 0], np.array(info["ue_start"])[None]))
check("D5", "the cell-free wanted signal is never weaker than the nearest-AP one (same channel)",
      np.all(d["signal_cellfree"] >= d["signal_nearest"] * (1 - 1e-12)))
check("D6", "cell-free: every AP uses exactly its whole budget",
      np.allclose(d["power_cellfree"].sum(axis=3), BUDGET))
ties = np.sum(np.abs(d["dist"][:, :, 0, :] - d["dist"][:, :, 1, :]) < 1e-9)
note(f"{ties} snapshots have a user exactly equally far from both APs (a tie, resolved in favour of AP0)")

print("\nE. THE RANDOM PART: does the flicker behave like Rayleigh fading?")
g = d["H"] / np.sqrt(d["beta"])[..., None]            # remove the path loss: what is left is the random part
n = g.size
power = np.abs(g) ** 2
check("E1", "random part: average 0, average power 1, real and imaginary variance 0.5",
      abs(g.mean()) < 0.025 and abs(power.mean() - 1) < 0.02 and abs(g.real.var() - 0.5) < 0.015 and abs(g.imag.var() - 0.5) < 0.015,
      f"mean {abs(g.mean()):.4f}, power {power.mean():.3f}, var {g.real.var():.3f}/{g.imag.var():.3f}")
x = np.sort(power.ravel())
ks = np.max(np.abs(np.arange(1, n + 1) / n - (1 - np.exp(-x))))
check("E2", "antenna powers follow the exponential law of Rayleigh fading (Kolmogorov-Smirnov test)",
      ks < 1.63 / np.sqrt(n), f"distance {ks:.4f}, limit {1.63/np.sqrt(n):.4f}")
corr_ant = np.corrcoef(g[..., 0].real.ravel(), g[..., 1].real.ravel())[0, 1]
check("E3", "different antennas are independent", abs(corr_ant) < 0.08, f"correlation {corr_ant:+.3f}")
corr_time = np.corrcoef(g[:, :-1].real.ravel(), g[:, 1:].real.ravel())[0, 1]
check("E4", "one second and the next one are independent (a new draw every second, as documented)",
      abs(corr_time) < 0.025, f"correlation {corr_time:+.3f}")
corr_pair = np.corrcoef(g[:, :, 0, 0, :].real.ravel(), g[:, :, 1, 2, :].real.ravel())[0, 1]
check("E5", "different AP-user pairs are independent", abs(corr_pair) < 0.06, f"correlation {corr_pair:+.3f}")
strength = np.sum(power, axis=4)
check("E6", f"average channel strength is {N} x beta (array gain)", abs(strength.mean() / N - 1) < 0.02,
      f"{strength.mean()/N:.3f}")
rel = strength.std() / strength.mean()
check("E7", f"channel hardening: relative randomness of the strength is 1/sqrt({N}) = {1/np.sqrt(N):.3f}",
      abs(rel / (1 / np.sqrt(N)) - 1) < 0.05, f"measured {rel:.3f}")

print("\nF. STRESS TESTS: does the answer change when it must not (and only then)?")
sample = [(e, t) for e in range(0, E, 3) for t in range(0, T, 8)]
p_half = max(np.max(np.abs(my_rates(d["H"][e, t], d["power_cellfree"][e, t] / 2)[3] - d["rate_cellfree"][e, t])
                    / d["rate_cellfree"][e, t]) for e, t in sample)
check("F1", "halving every power changes the speeds by less than 1% (noise is negligible)", p_half < 0.01, f"{p_half*100:.4f}%")
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
print(f"      {'item':24s} {'ours':>26s} {'paper':>20s}")
for name, (ours, theirs) in {**same, **paper}.items():
    print(f"      {name:24s} {str(ours):>26s} {str(theirs):>20s}   {'same' if str(ours) == str(theirs) or ours == theirs else 'DIFFERENT'}")
print(f"      all episodes start from the same 4 positions: {np.all(d['pos'][:, 0] == d['pos'][0, 0])}")

passed = sum(ok for _, ok in results)
print(f"\nAUDIT RESULT: {passed} of {len(results)} checks passed" + ("" if passed == len(results) else
      "  -> FAILED: " + ", ".join(code for code, ok in results if not ok)))