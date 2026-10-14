# The audit of the paper-size network (121 APs, 100 users, 64 antennas).
#   A  layout and the numbers of Table I
#   B  the serving rules at this size
#   C  the fast rate calculation against the library and against the audit's own loops
#   D  stress tests (halving the power is checked exactly, with and without noise)
#   E  the random part with 64 antennas
#   F  patterns carried over from the middle network (see the honest note in the printout)
# Saves tables/audit.csv.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the files of THIS folder always win
import csv
import time
import numpy as np

import paper_config as cfg
import paper_layout as layout
import paper_gains as gains
import paper_serving as serving
import paper_simulate as sim
import paper_paths as paths
from paper_precoding import mrt_precoders
from paper_rates import sinr_and_rate


records = []
B = cfg.B_HZ
NOISE_W_OWN = 10 ** ((cfg.N0_DBM_HZ + 10 * np.log10(B) + cfg.NF_DB - 30) / 10)      # the audit's own noise
BUDGET = 10 ** ((cfg.PTX_DBM - 30) / 10)


def check(code, group, text, ok, detail=""):
    ok = bool(ok)
    records.append([code, group, text, "PASS" if ok else "FAIL", detail])
    print(f"[{code}] {text}{' (' + detail + ')' if detail else ''} -> {'PASS' if ok else 'FAIL'}")


def spearman(a, b):
    return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]


def my_rates_for_users(H, p, users):
    """The audit's OWN calculation with explicit loops, for a few users (all 100 would take too long)."""
    M, K, N = H.shape
    W = np.zeros_like(H)
    for m in range(M):
        for j in range(K):
            if p[m, j] > 0:
                W[m, j] = np.sqrt(p[m, j]) * H[m, j] / np.linalg.norm(H[m, j])
    signal, rate = [], []
    for k in users:
        sig, itf = 0.0, 0.0
        for j in range(K):
            amp = sum(np.vdot(H[m, k], W[m, j]) for m in range(M))
            if j == k:
                sig = abs(amp) ** 2
            else:
                itf += abs(amp) ** 2
        signal.append(sig)
        rate.append(B * np.log2(1 + sig / (itf + NOISE_W_OWN)) / 1e6)
    return np.array(signal), np.array(rate)


ap_pos = layout.generate_ap_positions()
RULES = sim.standard_rules()
NEAREST, CLUSTER, EVERYONE = list(RULES)
L = cfg.L_PER_AP

# ---------------------------------------------------------------------
print("A. LAYOUT AND THE NUMBERS OF TABLE I")
check("A1", "table I", "121 APs, 100 users, 64 antennas, 1 km x 1 km, as in Table I of the paper",
      (cfg.N_AP, cfg.N_UE, cfg.N_ANTENNAS, cfg.AREA_M) == (121, 100, 64, 1000.0))
check("A2", "table I", "carrier 3.5 GHz, bandwidth 100 MHz, power 46 dBm, noise figure 9 dB, noise density -174 dBm/Hz",
      (cfg.FC_HZ, cfg.B_HZ, cfg.PTX_DBM, cfg.NF_DB, cfg.N0_DBM_HZ) == (3.5e9, 100e6, 46.0, 9.0, -174.0))
check("A3", "table I", "paper-like propagation: exponent 3.8 and shadowing 8 dB", cfg.PAPER_LIKE == dict(alpha=3.8, shadow_sigma_db=8.0))
grid = layout.grid_points()
shift = np.abs(ap_pos - grid).max()
dd = np.linalg.norm(ap_pos[:, None] - ap_pos[None], axis=2) + np.eye(cfg.N_AP) * 1e9
check("A4", "layout", f"each AP is within {cfg.AP_JITTER_M:.0f} m of its grid point, neighbours at least {cfg.AP_SPACING_M - 2 * cfg.AP_JITTER_M:.0f} m apart",
      shift <= cfg.AP_JITTER_M + 1e-9 and dd.min() >= cfg.AP_SPACING_M - 2 * cfg.AP_JITTER_M - 1e-9,
      f"largest shift {shift:.1f} m, closest pair {dd.min():.1f} m")
rng = np.random.default_rng(0)
drops = np.array([layout.generate_user_drop(rng) for _ in range(200)])
check("A5", "layout", "users are inside the map and spread evenly (average position close to the middle)",
      np.all((drops >= 0) & (drops <= cfg.AREA_M)) and np.all(np.abs(drops.mean(axis=(0, 1)) - cfg.AREA_M / 2) < 10),
      f"average position ({drops[..., 0].mean():.0f}, {drops[..., 1].mean():.0f}) m")

# ---------------------------------------------------------------------
print("\nB. THE SERVING RULES AT THIS SIZE (100 drops)")
rng = np.random.default_rng(3)
ok_near = ok_cl = ok_all = ok_pow = True
cover_all, aps_per_user, top_share = True, [], []
for _ in range(100):
    dist = gains.distance_matrix(ap_pos, layout.generate_user_drop(rng))
    beta = gains.beta_from_distance(dist, **cfg.PAPER_LIKE, rng=rng)
    mn, mc, me = serving.nearest_mask(beta), serving.cluster_mask(beta, L), serving.everyone_mask(beta)
    ok_near &= bool(np.all(mn.sum(axis=0) == 1) and np.array_equal(np.argmax(mn, axis=0), np.argmax(beta, axis=0)))
    ok_cl &= bool(np.all(mc.sum(axis=1) == L))
    cover_all &= bool(np.all(mc.any(axis=0)))
    aps_per_user.append(mc.sum(axis=0).mean())
    ok_all &= bool(me.all())
    top = np.zeros_like(mc)
    top[np.arange(cfg.N_AP)[:, None], np.argsort(-beta, axis=1)[:, :L]] = True
    top_share.append((mc & top).sum() / mc.sum())
    for mask in (mn, mc, me):
        used = serving.equal_power_per_ap(mask).sum(axis=1)
        ok_pow &= bool(np.allclose(used[mask.any(axis=1)], BUDGET) and np.all(used[~mask.any(axis=1)] == 0))
check("B1", "rules", "nearest rule: every user has exactly one AP, the one with the largest gain", ok_near)
check("B2", "rules", f"cluster rule: every AP serves exactly {L} users", ok_cl)
check("B3", "rules", "cluster rule: every one of the 100 users is served by at least one AP, in every drop", cover_all)
check("B4", "rules", f"cluster rule: on average every user hears {cfg.N_AP * L / cfg.N_UE:.2f} APs (= APs x L / users)",
      abs(np.mean(aps_per_user) - cfg.N_AP * L / cfg.N_UE) < 1e-9, f"{np.mean(aps_per_user):.2f}")
check("B5", "rules", "cluster rule: at least 90% of the beams go to the AP's own strongest users", np.mean(top_share) > 0.9,
      f"{np.mean(top_share) * 100:.1f}%")
check("B6", "rules", "everyone rule: every AP serves every user (12,100 beams)", ok_all)
check("B7", "rules", "power: an AP that sends uses exactly its whole budget, a silent AP uses nothing (all three rules)", ok_pow)

# ---------------------------------------------------------------------
print("\nC. THE RATE CALCULATION (2 drops x 3 rules)")
rng = np.random.default_rng(4)
worst_lib = worst_own = 0.0
for _ in range(2):
    drop = sim.simulate_drop(ap_pos, rng, RULES, **cfg.PAPER_LIKE)
    for r in drop["rules"].values():
        lib = sinr_and_rate(drop["H"], mrt_precoders(drop["H"], r["power"]))        # the Step 5 library (einsum)
        worst_lib = max(worst_lib, np.max(np.abs(lib[3] - r["rate"]) / r["rate"]))
        users = rng.choice(cfg.N_UE, 6, replace=False)
        _, own = my_rates_for_users(drop["H"], r["power"], users)                   # the audit's own loops
        worst_own = max(worst_own, np.max(np.abs(own - r["rate"][users]) / r["rate"][users]))
check("C1", "rates", "the fast calculation used at this size gives the same speeds as the Step 5 library", worst_lib < 1e-9,
      f"largest relative difference {worst_lib:.1e}")
check("C2", "rates", "speeds of 6 random users per drop recomputed with the audit's own loops", worst_own < 1e-8,
      f"largest relative difference {worst_own:.1e}")
start = time.perf_counter()
sim.simulate_drop(ap_pos, np.random.default_rng(9), RULES, **cfg.PAPER_LIKE)
seconds = time.perf_counter() - start
check("C3", "rates", "one drop (three rules, 64 antennas) takes less than 5 seconds", seconds < 5, f"{seconds:.2f} s")

# ---------------------------------------------------------------------
print("\nD. STRESS TESTS (3 drops)")
rng = np.random.default_rng(5)
noise_free = pred_err = change_max = change_mean = 0.0
perm_users = perm_aps = 0.0
ok_off = True
K_perm, M_perm = np.arange(cfg.N_UE)[::-1], np.random.default_rng(6).permutation(cfg.N_AP)
for _ in range(3):
    drop = sim.simulate_drop(ap_pos, rng, RULES, **cfg.PAPER_LIKE)
    r = drop["rules"][CLUSTER]
    H, p, rate = drop["H"], r["power"], r["rate"]
    rate_of = lambda H_, p_, noise=sim.NOISE_W: sim.fast_sinr_and_rate(H_, mrt_precoders(H_, p_), noise)
    half = rate_of(H, p / 2)[3]
    change = np.abs(half - rate) / rate
    change_max, change_mean = max(change_max, change.max()), max(change_mean, change.mean())
    # without noise the speed depends only on power RATIOS, so halving every power changes nothing
    noise_free = max(noise_free, np.max(np.abs(rate_of(H, p / 2, 0.0)[3] - rate_of(H, p, 0.0)[3]) / rate_of(H, p, 0.0)[3]))
    # with noise, halving the powers keeps signal / interference but doubles the weight of the noise
    predicted = B * np.log2(1 + r["signal"] / (r["interference"] + 2 * NOISE_W_OWN)) / 1e6
    pred_err = max(pred_err, np.max(np.abs(half - predicted) / predicted))
    perm_users = max(perm_users, np.max(np.abs(rate_of(H[:, K_perm], p[:, K_perm])[3] - rate[K_perm]) / rate[K_perm]))
    perm_aps = max(perm_aps, np.max(np.abs(rate_of(H[M_perm], p[M_perm])[3] - rate) / rate))
    p0 = p.copy()
    p0[:, 0] = 0.0
    s0, i0, _, rate0 = rate_of(H, p0)
    ok_off &= bool(rate0[0] == 0 and np.all(i0[1:] <= r["interference"][1:] * (1 + 1e-9)) and np.any(i0[1:] < r["interference"][1:]))
check("D1a", "stress", "without noise, halving every power changes the speeds by exactly nothing (only power ratios matter)",
      noise_free < 1e-9, f"{noise_free:.1e}")
check("D1b", "stress", "with noise, halving every power changes the speeds exactly as predicted by signal / (interference + 2 x noise)",
      pred_err < 1e-9, f"largest difference to the prediction {pred_err:.1e}")
print(f"      finding: with noise, halving every power changes the speed of the most affected user by {change_max * 100:.2f}% "
      f"(the average user by {change_mean * 100:.3f}%): noise matters a little for the weakest users with paper-like propagation")
check("D2", "stress", "renumbering the users only renumbers the results", perm_users < 1e-8, f"{perm_users:.1e}")
check("D3", "stress", "renumbering the APs changes nothing", perm_aps < 1e-8, f"{perm_aps:.1e}")
check("D4", "stress", "switching off the beams of one user: its speed is 0 and the others hear less interference", ok_off)

# ---------------------------------------------------------------------
print("\nE. THE RANDOM PART WITH 64 ANTENNAS (3 drops)")
rng = np.random.default_rng(7)
g_all = []
for _ in range(3):
    drop = sim.simulate_drop(ap_pos, rng, {"c": RULES[CLUSTER]}, **cfg.PAPER_LIKE)
    g_all.append(drop["H"] / np.sqrt(drop["beta"])[..., None])
g_all = np.array(g_all)
power = np.abs(g_all) ** 2
n = power.size
x = np.sort(power.ravel())
ks = np.max(np.abs(np.arange(1, n + 1) / n - (1 - np.exp(-x))))
check("E1", "random", "antenna powers follow the exponential law of Rayleigh fading (Kolmogorov-Smirnov)", ks < 1.63 / np.sqrt(n),
      f"{n:,} numbers, distance {ks:.5f}, limit {1.63 / np.sqrt(n):.5f}")
strength = power.sum(axis=3)
check("E2", "random", f"average channel strength is {cfg.N_ANTENNAS} x beta", abs(strength.mean() / cfg.N_ANTENNAS - 1) < 0.01,
      f"{strength.mean() / cfg.N_ANTENNAS:.4f}")
rel = strength.std() / strength.mean()
check("E3", "random", f"channel hardening: relative randomness of the strength is 1/sqrt({cfg.N_ANTENNAS}) = {1 / np.sqrt(cfg.N_ANTENNAS):.3f}",
      abs(rel / (1 / np.sqrt(cfg.N_ANTENNAS)) - 1) < 0.03, f"measured {rel:.3f}")
corr = np.corrcoef(g_all[:-1].real.ravel(), g_all[1:].real.ravel())[0, 1]
check("E4", "random", "one drop and the next one are independent", abs(corr) < 0.01, f"correlation {corr:+.4f}")

# ---------------------------------------------------------------------
print("\nF. PATTERNS CARRIED OVER FROM THE MIDDLE NETWORK")
print("   HONEST NOTE: I had already looked at the tables of run_tables.py before writing this group, so these are NOT blind predictions.")
print("   They are the patterns of the middle network, re-checked at the paper size. If one fails, it is a finding to discuss.")
res = {name: sim.simulate(ap_pos, RULES, **kw) for name, kw in (("free", cfg.FREE_SPACE), ("paper", cfg.PAPER_LIKE))}
rate = {k: {r: res[k]["rules"][r]["rate"] for r in RULES} for k in res}
rhos = {}
for k in res:
    rhos[k] = spearman(res[k]["dist_nearest"].ravel(), rate[k][NEAREST].ravel())
    check(f"F1{k[0]}", "pattern", f"conventional rule: the farther from its AP, the slower the user (rank correlation below -0.3, {k} propagation)",
          rhos[k] < -0.3, f"Spearman {rhos[k]:+.2f}")
n_users = res["paper"]["dist_nearest"].size
check("F1s", "pattern", "the same relation exists in both settings, even if weaker (rank correlation more than 4 standard errors below 0)",
      all(v < -4 / np.sqrt(n_users) for v in rhos.values()),
      f"free {rhos['free']:+.2f}, paper-like {rhos['paper']:+.2f}, standard error about {1 / np.sqrt(n_users):.3f}")
for k in res:
    check(f"F2{k[0]}", "pattern", f"clusters beat 'everyone serves everyone' in average speed ({k} propagation)",
          rate[k][CLUSTER].mean() > rate[k][EVERYONE].mean(), f"{rate[k][CLUSTER].mean():.0f} against {rate[k][EVERYONE].mean():.0f} Mbps")
for k in res:
    check(f"F3{k[0]}", "pattern", f"clusters serve the slowest 5% better than the conventional rule ({k} propagation)",
          np.percentile(rate[k][CLUSTER], 5) > np.percentile(rate[k][NEAREST], 5),
          f"p5 {np.percentile(rate[k][CLUSTER], 5):.0f} against {np.percentile(rate[k][NEAREST], 5):.0f} Mbps")
o = res["paper"]["rules"][CLUSTER]
no_noise = (B * np.log2(1 + o["signal"] / np.maximum(o["interference"], 1e-30)) / 1e6).mean()
check("F4", "pattern", "the network is limited by interference, not noise (no noise: change below 1%)",
      abs(no_noise / o["rate"].mean() - 1) < 0.01, f"{abs(no_noise / o['rate'].mean() - 1) * 100:.3f}%")
p5 = {}
for LL in (1, 4, 7, 10):
    p5[LL] = np.percentile(sim.simulate(ap_pos, {"c": (lambda b, LL=LL: serving.cluster_mask(b, LL))}, **cfg.PAPER_LIKE)["rules"]["c"]["rate"], 5)
check("F5", "pattern", "paper-like propagation: the slowest 5% do better with L = 4 than with L = 1, with L = 7 and with L = 10",
      p5[4] > p5[1] and p5[4] > p5[7] and p5[4] > p5[10],
      f"p5: L=1 {p5[1]:.0f}, L=4 {p5[4]:.0f}, L=7 {p5[7]:.0f}, L=10 {p5[10]:.0f} Mbps")
middle_path = paths.find_middle_results()
if middle_path:
    with open(middle_path, encoding="utf-8") as f:
        middle = {(r["propagation"], r["rule"]): r for r in csv.DictReader(f)}
    key = ("paper-like (alpha 3.8, shadowing 8 dB)", f"clusters (L={L})")
    same_n = sim.simulate(ap_pos, {"c": RULES[CLUSTER]}, n_ant=16, **cfg.PAPER_LIKE)["rules"]["c"]["rate"]
    mid_mean = float(middle[key]["mean_Mbps"])
    check("F6", "pattern", "with the same 16 antennas, the paper-size network gives about the same cluster speed as the middle network (within 10%)",
          abs(same_n.mean() / mid_mean - 1) < 0.10, f"{same_n.mean():.0f} against {mid_mean:.0f} Mbps")
else:
    print("      F6 skipped: step8a_results.csv not found in cellfree_project (run step8a_run.py there first)")
again = sim.simulate(ap_pos, RULES, seed=cfg.SEED_DROPS, **cfg.FREE_SPACE)
check("F7", "pattern", "running the simulation again with the same seed gives identical numbers",
      all(np.array_equal(again["rules"][r]["rate"], res["free"]["rules"][r]["rate"]) for r in RULES))

code_records = [r for r in records if r[0][0] in "ABCDE"]
pattern_records = [r for r in records if r[0][0] == "F"]
print(f"\nCHECKS OF THE CODE AND THE DATA (groups A to E): {sum(r[3] == 'PASS' for r in code_records)} of {len(code_records)} passed"
      + "".join(f"  -> FAILED: {r[0]}" for r in code_records if r[3] == "FAIL"))
print(f"PATTERNS CARRIED OVER (group F): {sum(r[3] == 'PASS' for r in pattern_records)} of {len(pattern_records)} held"
      + ("" if all(r[3] == "PASS" for r in pattern_records) else
         "  -> to discuss (findings, not bugs): " + ", ".join(r[0] for r in pattern_records if r[3] == "FAIL")))
with open(paths.table("audit.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["code", "group", "check", "result", "detail"])
    w.writerows(records)
print("Saved tables/audit.csv")