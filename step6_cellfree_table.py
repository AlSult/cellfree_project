# Cell-free: EVERY AP serves EVERY user. Compared with Step 5 (nearest AP only).

import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import full_channel
from power_allocation import nearest_ap_mask, cooperative_mask, equal_power_allocation
from precoding import mrt_precoders, signal_seen_by_users
from rate_calculation import sinr_and_rate, NOISE_W

np.set_printoptions(linewidth=150, suppress=True)
n_ap, n_ue = config.N_AP, config.N_UE
idx = np.arange(n_ue)

dist, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
rules = {"nearest": nearest_ap_mask(dist), "cellfree": cooperative_mask()}
power = {name: equal_power_allocation(mask) for name, mask in rules.items()}
budget = 10 ** ((config.PTX_DBM - 30) / 10)

# ---------------- the seed-0 draw: the same random channel as in Steps 3 to 5 ----------------
H = full_channel(beta, np.random.default_rng(config.SEED))
res = {name: sinr_and_rate(H, mrt_precoders(H, power[name])) for name in rules}

print("Power used by each AP (W), budget is %.1f W:" % budget)
for name in rules:
    print(f"   {name:9s} AP0 {power[name][0].sum():5.1f}   AP1 {power[name][1].sum():5.1f}   total beams {rules[name].sum()}")

print("\nTHE SEED-0 DRAW")
print(f"{'user':>5} | {'Step 5 (nearest AP)':>26} | {'Cell-free (both APs)':>26} | {'change':>8}")
print(f"{'':>5} | {'SINR (dB)':>12} {'speed (Mbps)':>13} | {'SINR (dB)':>12} {'speed (Mbps)':>13} | {'':>8}")
for k in range(n_ue):
    s5, sc = res["nearest"][2][k], res["cellfree"][2][k]
    r5, rc = res["nearest"][3][k], res["cellfree"][3][k]
    print(f"{k:5d} | {10*np.log10(s5):12.1f} {r5:13.1f} | {10*np.log10(sc):12.1f} {rc:13.1f} | {100*(rc-r5)/r5:+7.0f}%")

# ---------------- CHECKS ----------------
print("\nCHECKS")

# 1. Every AP sends 4 beams and uses exactly its power budget
used = power["cellfree"].sum(axis=1)
ok1 = bool(rules["cellfree"].sum() == n_ap * n_ue and np.allclose(used, budget))
print(f"[1] cell-free: {rules['cellfree'].sum()} beams, every AP uses exactly its budget ({used.round(1)} W of {budget:.1f} W) -> {'PASS' if ok1 else 'FAIL'}")

# 2. The wanted signal is a COHERENT sum: the AMPLITUDES of the two APs add up
A = signal_seen_by_users(H, mrt_precoders(H, power["cellfree"]))
amp = np.abs(A[:, idx, idx])                       # (AP, user): amplitude each AP contributes to its wanted signal
coherent = amp.sum(axis=0) ** 2                    # amplitudes add, then square
incoherent = (amp ** 2).sum(axis=0)                # what we would get if only the POWERS added
ok2 = bool(np.allclose(coherent, res["cellfree"][0]) and np.all(coherent >= incoherent))
print(f"[2] wanted signal = (sum of the AMPLITUDES of both APs)^2, never less than the sum of powers -> {'PASS' if ok2 else 'FAIL'}")
print(f"    extra gain of combining over adding powers, per user (dB): {(10*np.log10(coherent/incoherent)).round(2)}")

# 3. A slow, one-beam-at-a-time calculation gives the same signal and interference
W = mrt_precoders(H, power["cellfree"])
sig2, int2 = np.zeros(n_ue), np.zeros(n_ue)
for k in range(n_ue):
    for j in range(n_ue):
        received = sum(np.vdot(H[m, k], W[m, j]) for m in range(n_ap))
        if j == k:
            sig2[k] = abs(received) ** 2
        else:
            int2[k] += abs(received) ** 2
ok3 = bool(np.allclose(sig2, res["cellfree"][0]) and np.allclose(int2, res["cellfree"][1]))
print(f"[3] signal and interference agree with the slow one-beam-at-a-time calculation -> {'PASS' if ok3 else 'FAIL'}")

# 4. Same speeds if every power is halved: the noise is negligible, so only the ratios matter
p_half = {name: power[name] / 2 for name in rules}
r_half = sinr_and_rate(H, mrt_precoders(H, p_half["cellfree"]))[3]
diff = np.max(np.abs(r_half - res["cellfree"][3]) / res["cellfree"][3])
ok4 = diff < 0.01
print(f"[4] halving ALL powers changes the speeds by at most {diff*100:.3f}% -> {'PASS' if ok4 else 'FAIL'}"
      f"  (so the comparison is not a trick of more total power)")

# ---------------- many draws, the SAME random channels for both rules ----------------
trials = 10000
rng = np.random.default_rng(100)
rates = {name: np.zeros((trials, n_ue)) for name in rules}
interf_sum = {name: np.zeros(n_ue) for name in rules}
for t in range(trials):
    Ht = full_channel(beta, rng)
    for name in rules:
        _, it, _, rt = sinr_and_rate(Ht, mrt_precoders(Ht, power[name]))
        rates[name][t] = rt
        interf_sum[name] += it

# 5. average interference = sum over the other beams and over the APs of p x beta
ok5 = True
worst = 0.0
for name in rules:
    expected = np.array([sum(power[name][m, j] * beta[m, k] for j in range(n_ue) if j != k for m in range(n_ap))
                         for k in range(n_ue)])
    err = np.abs(interf_sum[name] / trials - expected) / expected
    worst = max(worst, err.max())
    ok5 &= bool(err.max() < 0.05)
print(f"[5] average interference over {trials} draws vs the formula, both rules: largest error {worst*100:.1f}% -> {'PASS' if ok5 else 'FAIL'}")

# 6. One lonely user (no other beams): cooperation can never hurt, and at the exact middle
#    between two equal APs it must give exactly +6.02 dB (two equal amplitudes: (2a)^2 = 4 a^2)
grid = np.linspace(0, config.AREA_M, 201)
Xg, Yg = np.meshgrid(grid, grid)
ap_pos = generate_ap_positions()
d_g = np.maximum(np.stack([np.hypot(Xg - ap_pos[m, 0], Yg - ap_pos[m, 1]) for m in range(n_ap)]), 1.0)
beta_g = 10 ** (-(20 * np.log10(4 * np.pi * d_g * config.FC_HZ / config.C)) / 10)
amp_g = np.sqrt(budget / n_ue * config.N_ANTENNAS * beta_g)      # amplitude each AP gives (average)
gain_db = 10 * np.log10(amp_g.sum(axis=0) ** 2 / amp_g.max(axis=0) ** 2)
mid = np.argmin(np.abs(grid - 100))
ok6 = bool(np.all(gain_db >= -1e-9) and abs(gain_db[mid, mid] - 20 * np.log10(2)) < 1e-6)
print(f"[6] lone user: cooperation never hurts on a 201x201 grid; gain at the exact middle {gain_db[mid, mid]:.2f} dB "
      f"(expected 6.02) -> {'PASS' if ok6 else 'FAIL'}")

# ---------------- the averages ----------------
avg = {name: rates[name].mean(axis=0) for name in rules}
p5 = {name: np.percentile(rates[name], 5, axis=0) for name in rules}
print(f"\nAFTER {trials} DRAWS (both rules see exactly the same random channels)")
print(f"{'user':>5} | {'Step 5: average':>16} {'5% worst':>10} | {'Cell-free: average':>19} {'5% worst':>10} | {'change':>8}")
for k in range(n_ue):
    print(f"{k:5d} | {avg['nearest'][k]:12.0f} Mbps {p5['nearest'][k]:7.0f} | {avg['cellfree'][k]:15.0f} Mbps {p5['cellfree'][k]:7.0f} | "
          f"{100*(avg['cellfree'][k]-avg['nearest'][k])/avg['nearest'][k]:+7.0f}%")
slowest = {name: rates[name].min(axis=1) for name in rules}
print(f"\n{'':>24} {'Step 5':>10} {'Cell-free':>10}")
print(f"{'sum of the 4 averages':>24} {avg['nearest'].sum():7.0f} Mbps {avg['cellfree'].sum():7.0f} Mbps")
print(f"{'slowest user (average)':>24} {avg['nearest'].min():7.0f} Mbps {avg['cellfree'].min():7.0f} Mbps")
print(f"{'slowest user per draw':>24} {slowest['nearest'].mean():7.0f} Mbps {slowest['cellfree'].mean():7.0f} Mbps   (average over draws)")

# ---------------- back-of-the-envelope for cell-free ----------------
print("\nBACK-OF-THE-ENVELOPE (cell-free): SINR ~ 16 x (sum over APs of sqrt(beta))^2 / (3 x sum over APs of beta)")
print(f"{'user':>5} | {'estimated':>10} {'simulated':>10} {'difference':>11}")
for k in range(n_ue):
    est_sinr = config.N_ANTENNAS * np.sum(np.sqrt(beta[:, k])) ** 2 / ((n_ue - 1) * np.sum(beta[:, k]))
    est = config.B_HZ * np.log2(1 + est_sinr) / 1e6
    print(f"{k:5d} | {est:7.0f} Mbps {avg['cellfree'][k]:6.0f} Mbps {100*(est-avg['cellfree'][k])/avg['cellfree'][k]:+10.0f}%")

#np.savetxt("step6_average_speeds.csv", np.column_stack([avg["nearest"], avg["cellfree"]]), delimiter=",", fmt="%.2f",
#           header="step5_nearest_Mbps,step6_cellfree_Mbps")
#print("\nSaved step6_average_speeds.csv")