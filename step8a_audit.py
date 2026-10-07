# The audit of the middle network (25 APs, 20 users). Same spirit as the audit of the tiny
# network: independent recomputation, stress tests, statistics. New in this audit:
#   - checks of the serving rules (clusters)
#   - EXPECTATIONS and OBSERVED PATTERNS (group G): what the tiny network and the physics lead
#     us to expect, plus patterns seen in a first trial run (marked as such). If one fails,
#     it is a finding to discuss, not necessarily a bug.

import csv
import numpy as np

import config
import mid_config as cfg
import mid_layout
import mid_gains
import mid_serving
import mid_simulate as sim

records = []
B = cfg.B_HZ
NOISE_W_OWN = 10 ** ((cfg.N0_DBM_HZ + 10 * np.log10(B) + cfg.NF_DB - 30) / 10)      # the audit's own noise
BUDGET = 10 ** ((cfg.PTX_DBM - 30) / 10)


def check(code, group, text, ok, detail=""):
    ok = bool(ok)
    records.append([code, group, text, "PASS" if ok else "FAIL", detail])
    print(f"[{code}] {text}{' (' + detail + ')' if detail else ''} -> {'PASS' if ok else 'FAIL'}")


def note(text):
    print(f"      finding: {text}")


def my_rates(H, p):
    """The audit's OWN calculation, with explicit loops (no shared code with the library)."""
    M, K, N = H.shape
    W = np.zeros_like(H)
    for m in range(M):
        for j in range(K):
            if p[m, j] > 0:
                W[m, j] = np.sqrt(p[m, j]) * H[m, j] / np.linalg.norm(H[m, j])
    sig, itf = np.zeros(K), np.zeros(K)
    for k in range(K):
        for j in range(K):
            amp = sum(np.vdot(H[m, k], W[m, j]) for m in range(M))
            if j == k:
                sig[k] = abs(amp) ** 2
            else:
                itf[k] += abs(amp) ** 2
    sinr = sig / (itf + NOISE_W_OWN)
    return sig, itf, B * np.log2(1 + sinr) / 1e6


def spearman(a, b):
    return np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]


ap_pos = mid_layout.generate_ap_positions()
RULES = sim.standard_rules()
NEAREST, CLUSTER, EVERYONE = list(RULES)
L = cfg.L_PER_AP

# ---------------------------------------------------------------------
print("A. LAYOUT")
grid = mid_layout.grid_points()
shift = np.abs(ap_pos - grid).max()
dd = np.linalg.norm(ap_pos[:, None] - ap_pos[None], axis=2) + np.eye(cfg.N_AP) * 1e9
check("A1", "layout", f"{cfg.N_AP} APs, each within {cfg.AP_JITTER_M:.0f} m of its grid point, neighbours at least 80 m apart",
      ap_pos.shape == (cfg.N_AP, 2) and shift <= cfg.AP_JITTER_M + 1e-9 and dd.min() >= 80 - 1e-9,
      f"largest shift {shift:.1f} m, closest pair {dd.min():.1f} m")
rng = np.random.default_rng(0)
drops = np.array([mid_layout.generate_user_drop(rng) for _ in range(300)])
check("A2", "layout", "users are inside the map and spread evenly (average position close to the middle)",
      np.all((drops >= 0) & (drops <= cfg.AREA_M)) and np.all(np.abs(drops.mean(axis=(0, 1)) - cfg.AREA_M / 2) < 8),
      f"average position ({drops[..., 0].mean():.0f}, {drops[..., 1].mean():.0f}) m")

# ---------------------------------------------------------------------
print("\nB. THE CHANNEL GAINS (new code that must agree with the old one)")
from path_loss_model import distance_table, beta_from_positions
from ap_location_generation import generate_ap_positions as tiny_ap
from ue_location_generation import generate_ue_positions as tiny_ue
mine = mid_gains.beta_from_distance(distance_table(tiny_ap(), tiny_ue()))
check("B1", "gains", "on the tiny layout the new gain function gives exactly the Step 2 numbers",
      np.allclose(mine, beta_from_positions(tiny_ap(), tiny_ue())[2], rtol=1e-12))
d_test = np.random.default_rng(1).uniform(2, 400, 500)
loss = -10 * np.log10(mid_gains.beta_from_distance(d_test))
rule_of_thumb = 32.44 + 20 * np.log10(cfg.FC_HZ / 1e6) + 20 * np.log10(d_test / 1000)
check("B2", "gains", "free-space loss agrees with the engineers' rule of thumb", np.max(np.abs(loss - rule_of_thumb)) < 0.01,
      f"largest difference {np.max(np.abs(loss - rule_of_thumb)):.4f} dB")
d_grid = np.logspace(0.3, 2.5, 60)
loss38 = -10 * np.log10(mid_gains.beta_from_distance(d_grid, alpha=3.8))
slope = np.polyfit(np.log10(d_grid), loss38, 1)[0]
check("B3", "gains", "with alpha = 3.8 the loss grows 38 dB for every 10 times the distance", abs(slope - 38) < 0.01,
      f"slope {slope:.2f} dB per decade")
d50 = np.full(20000, 50.0)
with_shadow = -10 * np.log10(mid_gains.beta_from_distance(d50, 3.8, 8.0, np.random.default_rng(2)))
without_shadow = -10 * np.log10(mid_gains.beta_from_distance(d50[:1], 3.8)[0])
shadow = with_shadow - without_shadow                     # the extra loss caused by shadowing alone, in dB
check("B4", "gains", "shadowing: average 0 dB and spread 8 dB around the path loss",
      abs(shadow.mean()) < 0.2 and abs(shadow.std() - 8) < 0.2, f"average {shadow.mean():+.2f} dB, spread {shadow.std():.2f} dB")

# ---------------------------------------------------------------------
print("\nC. THE SERVING RULES (300 drops)")
rng = np.random.default_rng(3)
ok_near = ok_cl = ok_all = ok_pow = True
cover_all, aps_per_user, top_share = True, [], []
for _ in range(300):
    dist = mid_gains.distance_matrix(ap_pos, mid_layout.generate_user_drop(rng))
    beta = mid_gains.beta_from_distance(dist)
    mn, mc, me = mid_serving.nearest_mask(beta), mid_serving.cluster_mask(beta, L), mid_serving.everyone_mask(beta)
    ok_near &= bool(np.all(mn.sum(axis=0) == 1) and np.array_equal(np.argmax(mn, axis=0), np.argmin(dist, axis=0)))
    ok_cl &= bool(np.all(mc.sum(axis=1) == L))
    cover_all &= bool(np.all(mc.any(axis=0)))
    aps_per_user.append(mc.sum(axis=0).mean())
    ok_all &= bool(me.all())
    top = np.zeros_like(mc)
    top[np.arange(cfg.N_AP)[:, None], np.argsort(-beta, axis=1)[:, :L]] = True
    top_share.append((mc & top).sum() / mc.sum())
    for mask in (mn, mc, me):
        p = mid_serving.equal_power_per_ap(mask)
        used = p.sum(axis=1)
        ok_pow &= bool(np.allclose(used[mask.any(axis=1)], BUDGET) and np.all(used[~mask.any(axis=1)] == 0))
check("C1", "rules", "nearest rule: every user has exactly one AP, and it is the closest one", ok_near)
check("C2", "rules", f"cluster rule: every AP serves exactly {L} users", ok_cl)
check("C3", "rules", "cluster rule: every user is served by at least one AP, in every drop", cover_all)
check("C4", "rules", f"cluster rule: on average every user hears {cfg.N_AP * L / cfg.N_UE:.2f} APs (= APs x L / users)",
      abs(np.mean(aps_per_user) - cfg.N_AP * L / cfg.N_UE) < 1e-9, f"{np.mean(aps_per_user):.2f}")
check("C5", "rules", "cluster rule: at least 90% of the beams go to the AP's own strongest users (the rest come from the coverage fix)",
      np.mean(top_share) > 0.9, f"{np.mean(top_share)*100:.1f}%")
check("C6", "rules", "everyone rule: every AP serves every user", ok_all)
check("C7", "rules", "power: an AP that sends uses exactly its whole budget, a silent AP uses nothing (all three rules)", ok_pow)

# ---------------------------------------------------------------------
print("\nD. INDEPENDENT RECOMPUTATION (10 drops x 3 rules x 2 kinds of propagation)")
worst = 0.0
for kw in (cfg.FREE_SPACE, cfg.PAPER_LIKE):
    rng = np.random.default_rng(4)
    for _ in range(10):
        drop = sim.simulate_drop(ap_pos, rng, RULES, kw["alpha"], kw["shadow_sigma_db"])
        for r in drop["rules"].values():
            s, i, rate = my_rates(drop["H"], r["power"])
            worst = max(worst, np.max(np.abs(rate - r["rate"]) / r["rate"]), np.max(np.abs(s - r["signal"]) / r["signal"]))
check("D1", "recomputation", "speeds and wanted signals recomputed with the audit's own loops", worst < 1e-8,
      f"largest relative difference {worst:.1e}")

# ---------------------------------------------------------------------
print("\nE. STRESS TESTS (10 drops)")
rng = np.random.default_rng(5)
half = perm_users = perm_aps = 0.0
ok_off = True
K_perm, M_perm = np.arange(cfg.N_UE)[::-1], np.random.default_rng(6).permutation(cfg.N_AP)
for _ in range(10):
    drop = sim.simulate_drop(ap_pos, rng, RULES)
    r = drop["rules"][CLUSTER]
    H, p, rate = drop["H"], r["power"], r["rate"]
    half = max(half, np.max(np.abs(my_rates(H, p / 2)[2] - rate) / rate))
    perm_users = max(perm_users, np.max(np.abs(my_rates(H[:, K_perm], p[:, K_perm])[2] - rate[K_perm]) / rate[K_perm]))
    perm_aps = max(perm_aps, np.max(np.abs(my_rates(H[M_perm], p[M_perm])[2] - rate) / rate))
    p0 = p.copy()
    p0[:, 0] = 0.0
    s, i, rate0 = my_rates(H, p0)
    ok_off &= bool(rate0[0] == 0 and np.all(i[1:] <= r["interference"][1:] * (1 + 1e-12)) and np.any(i[1:] < r["interference"][1:]))
check("E1", "stress", "halving every power changes the speeds by less than 1%", half < 0.01, f"{half*100:.4f}%")
check("E2", "stress", "renumbering the users only renumbers the results", perm_users < 1e-8, f"{perm_users:.1e}")
check("E3", "stress", "renumbering the APs changes nothing", perm_aps < 1e-8, f"{perm_aps:.1e}")
check("E4", "stress", "switching off the beams of one user: its speed is 0 and the others hear less interference", ok_off)

# ---------------------------------------------------------------------
print("\nF. THE RANDOM PART (50 drops)")
rng = np.random.default_rng(7)
g_all = []
for _ in range(50):
    drop = sim.simulate_drop(ap_pos, rng, {"c": RULES[CLUSTER]})
    g_all.append(drop["H"] / np.sqrt(drop["beta"])[..., None])
g_all = np.array(g_all)
power = np.abs(g_all) ** 2
n = power.size
x = np.sort(power.ravel())
ks = np.max(np.abs(np.arange(1, n + 1) / n - (1 - np.exp(-x))))
check("F1", "random", "antenna powers follow the exponential law of Rayleigh fading (Kolmogorov-Smirnov)", ks < 1.63 / np.sqrt(n),
      f"distance {ks:.4f}, limit {1.63/np.sqrt(n):.4f}")
check("F2", "random", f"average channel strength is {config.N_ANTENNAS} x beta", abs(power.sum(axis=3).mean() / config.N_ANTENNAS - 1) < 0.02,
      f"{power.sum(axis=3).mean() / config.N_ANTENNAS:.3f}")
corr = np.corrcoef(g_all[:-1].real.ravel(), g_all[1:].real.ravel())[0, 1]
check("F3", "random", "one drop and the next one are independent", abs(corr) < 0.01, f"correlation {corr:+.4f}")

# ---------------------------------------------------------------------
print("\nG. EXPECTATIONS AND OBSERVED PATTERNS")
print("   HONEST NOTE: I ran a quick trial of the middle network BEFORE writing this group, so these are not blind predictions.")
print("   G1, G2f, G3f and G4 follow from the tiny network and from the physics (they were expected before the trial).")
print("   G2p, G3p and G5 are patterns I saw in the trial and turned into checks.")
print("   A failure here is a finding to discuss, not necessarily a bug.")
res = {name: sim.simulate(ap_pos, RULES, **kw) for name, kw in (("free", cfg.FREE_SPACE), ("paper", cfg.PAPER_LIKE))}
rate = {n: {r: res[n]["rules"][r]["rate"] for r in RULES} for n in res}
rho = spearman(res["free"]["dist_nearest"].ravel(), rate["free"][NEAREST].ravel())
check("G1", "expectation", "conventional rule: the farther from its AP, the slower the user (rank correlation below -0.3)", rho < -0.3,
      f"Spearman {rho:+.2f}")
for n in res:
    check(f"G2{n[0]}", "expectation" if n == "free" else "observed pattern", f"clusters beat 'everyone serves everyone' in average speed ({n} propagation)",
          rate[n][CLUSTER].mean() > rate[n][EVERYONE].mean(), f"{rate[n][CLUSTER].mean():.0f} against {rate[n][EVERYONE].mean():.0f} Mbps")
for n in res:
    check(f"G3{n[0]}", "expectation" if n == "free" else "observed pattern", f"clusters serve the slowest 5% better than the conventional rule ({n} propagation)",
          np.percentile(rate[n][CLUSTER], 5) > np.percentile(rate[n][NEAREST], 5),
          f"p5 {np.percentile(rate[n][CLUSTER], 5):.0f} against {np.percentile(rate[n][NEAREST], 5):.0f} Mbps")
o = res["free"]["rules"][CLUSTER]
no_noise = (B * np.log2(1 + o["signal"] / np.maximum(o["interference"], 1e-30)) / 1e6).mean()
check("G4", "expectation", "the network is limited by interference, not noise (no noise: change below 1%)",
      abs(no_noise / o["rate"].mean() - 1) < 0.01, f"{abs(no_noise / o['rate'].mean() - 1) * 100:.3f}%")
for n, kw in (("free", cfg.FREE_SPACE), ("paper", cfg.PAPER_LIKE)):
    p5 = {}
    for LL in (1, L, 20):
        p5[LL] = np.percentile(sim.simulate(ap_pos, {"c": (lambda b, LL=LL: mid_serving.cluster_mask(b, LL))}, **kw)["rules"]["c"]["rate"], 5)
    check(f"G5{n[0]}", "observed pattern", f"the slowest 5% do best with a few users per AP, not with 1 and not with everyone ({n} propagation)",
          p5[L] > p5[1] and p5[L] > p5[20], f"p5: L=1 {p5[1]:.0f}, L={L} {p5[L]:.0f}, L=20 {p5[20]:.0f} Mbps")
again = sim.simulate(ap_pos, RULES, n_drops=300, seed=cfg.SEED_DROPS)
check("G6", "expectation", "running the simulation again with the same seed gives identical numbers",
      all(np.array_equal(again["rules"][r]["rate"], res["free"]["rules"][r]["rate"]) for r in RULES))

print("\n   FINDINGS that are not pass or fail:")
note(f"with paper-like propagation the conventional rule has the HIGHER average speed "
     f"({rate['paper'][NEAREST].mean():.0f} against {rate['paper'][CLUSTER].mean():.0f} Mbps for clusters), "
     f"but the clusters keep the better slowest 5% and a much smaller spread")
note(f"{100 * np.mean(res['free']['rules'][NEAREST]['users_per_ap'] == 0):.0f}% of the APs serve nobody under the conventional rule "
     f"(only {cfg.N_UE / cfg.N_AP:.2f} users per AP)")

passed = sum(r[3] == "PASS" for r in records)
print(f"\nAUDIT RESULT: {passed} of {len(records)} checks passed" +
      ("" if passed == len(records) else "  -> FAILED: " + ", ".join(r[0] for r in records if r[3] == "FAIL")))
with open("step8a_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["code", "group", "check", "result", "detail"])
    w.writerows(records)
print("Saved step8a_audit.csv")