# Noise, interference, SINR and speed (Mbps) for our 2 APs x 4 users.
# Every user is served by its nearest AP only. Everything is checked.

import math
import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import full_channel
from power_allocation import nearest_ap_mask, equal_power_allocation
from precoding import mrt_precoders, signal_seen_by_users
from rate_calculation import sinr_and_rate, noise_power_dbm, NOISE_W

np.set_printoptions(linewidth=150, suppress=True)
n_ap, n_ue = config.N_AP, config.N_UE


def dbm(x):
    """Watts -> dBm"""
    return 10 * np.log10(x) + 30


dist, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
mask = nearest_ap_mask(dist)
p = equal_power_allocation(mask)
serving_ap = np.argmax(mask, axis=0)

# The same random channel as in Steps 3 and 4 (seed 0)
H = full_channel(beta, np.random.default_rng(config.SEED))
W = mrt_precoders(H, p)
signal, interference, sinr, rate = sinr_and_rate(H, W)
rate_no_interf = config.B_HZ * np.log2(1 + signal / NOISE_W) / 1e6      # as if nobody else existed
rate_no_noise = config.B_HZ * np.log2(1 + signal / interference) / 1e6  # as if there were no noise

print(f"Noise floor of the receiver: {noise_power_dbm():.1f} dBm")
print(f"Power of every beam: {dbm(p[mask][0]):.1f} dBm  ({p[mask][0]:.2f} W)")
print(f"Serving AP of each user: {serving_ap}   (user 1 is equally far from both, AP0 wins the tie)\n")

print(f"{'user':>5} {'AP':>3} | {'signal':>9} {'interference':>13} {'noise':>8} | {'SINR':>8} | {'speed':>10} | {'no interf.':>11} {'no noise':>10}")
print(f"{'':>5} {'':>3} | {'(dBm)':>9} {'(dBm)':>13} {'(dBm)':>8} | {'(dB)':>8} | {'(Mbps)':>10} | {'(Mbps)':>11} {'(Mbps)':>10}")
for k in range(n_ue):
    print(f"{k:5d} {serving_ap[k]:3d} | {dbm(signal[k]):9.1f} {dbm(interference[k]):13.1f} {noise_power_dbm():8.1f} | "
          f"{10*np.log10(sinr[k]):8.1f} | {rate[k]:10.1f} | {rate_no_interf[k]:11.1f} {rate_no_noise[k]:10.1f}")
print(f"\ntotal speed of the network (sum over the 4 users): {rate.sum():.0f} Mbps")

#np.savetxt("step5_rates.csv", np.column_stack([serving_ap, dbm(signal), dbm(interference), 10*np.log10(sinr), rate]),
#           delimiter=",", fmt="%.3f", header="serving_ap,signal_dBm,interference_dBm,sinr_dB,rate_Mbps")
#print("Saved step5_rates.csv")

# ---------------- CHECKS ----------------
print("\nCHECKS")

# 1. Noise floor by hand: -174 dBm/Hz + 10*log10(100e6 Hz) + 9 dB = -174 + 80 + 9 = -85 dBm
ok1 = abs(noise_power_dbm() - (-174 + 80 + 9)) < 1e-9
print(f"[1] noise floor: code {noise_power_dbm():.1f} dBm, by hand -174 + 80 + 9 = -85 dBm -> {'PASS' if ok1 else 'FAIL'}")

# 2. Nobody uses more power than the budget, and every user is served by exactly one AP
used = p.sum(axis=1)
budget = 10 ** ((config.PTX_DBM - 30) / 10)
ok2 = bool(np.all(used <= budget + 1e-9) and np.all(mask.sum(axis=0) == 1))
print(f"[2] power used per AP {used.round(1)} W, budget {budget:.1f} W; each user has exactly one AP -> {'PASS' if ok2 else 'FAIL'}")

# 3. Compute signal and interference a second, slower way (explicit loops, one beam at a time)
sig2, int2 = np.zeros(n_ue), np.zeros(n_ue)
for k in range(n_ue):
    for j in range(n_ue):
        received = sum(np.vdot(H[m, k], W[m, j]) for m in range(n_ap))   # sum over the APs
        if j == k:
            sig2[k] = abs(received) ** 2
        else:
            int2[k] += abs(received) ** 2
ok3 = bool(np.allclose(sig2, signal) and np.allclose(int2, interference))
print(f"[3] signal and interference agree with a slow one-beam-at-a-time calculation -> {'PASS' if ok3 else 'FAIL'}")

# 4. Shannon rate recomputed from the SINR in dB
rate2 = config.B_HZ * np.log2(1 + 10 ** (10 * np.log10(sinr) / 10)) / 1e6
ok4 = bool(np.allclose(rate, rate2))
print(f"[4] speed = bandwidth x log2(1 + SINR) recomputed from the SINR in dB -> {'PASS' if ok4 else 'FAIL'}")

# 5. Interference and noise can only slow a user down, never speed it up
ok5 = bool(np.all(rate <= rate_no_interf + 1e-9) and np.all(rate <= rate_no_noise + 1e-9))
print(f"[5] with interference AND noise every user is slower than with only one of them -> {'PASS' if ok5 else 'FAIL'}")

# 6. Many draws: the average interference must match a simple formula.
#    From Step 4: a beam meant for user j is heard by user k with, on average, power
#    p(j) x beta(AP serving j -> k)   (the same as ONE antenna would give).
expected_interf = np.zeros(n_ue)
for k in range(n_ue):
    for j in range(n_ue):
        if j != k:
            expected_interf[k] += p[serving_ap[j], j] * beta[serving_ap[j], k]

trials = 10000
rng = np.random.default_rng(100)
interf_sum = np.zeros(n_ue)
rates = np.zeros((trials, n_ue))
for t in range(trials):
    Ht = full_channel(beta, rng)
    Wt = mrt_precoders(Ht, p)
    _, it, _, rates[t] = sinr_and_rate(Ht, Wt)
    interf_sum += it
interf_avg = interf_sum / trials
err = np.abs(interf_avg - expected_interf) / expected_interf
ok6 = err.max() < 0.05
print(f"[6] average interference over {trials} draws vs the formula: largest error {err.max()*100:.1f}% -> {'PASS' if ok6 else 'FAIL'}")

print(f"\nAFTER {trials} DRAWS (each draw = one new random channel)")
print(f"{'user':>5} | {'average speed':>14} | {'5% worst draws are below':>25} | {'best 5% are above':>18}")
for k in range(n_ue):
    print(f"{k:5d} | {rates[:, k].mean():11.1f} Mbps | {np.percentile(rates[:, k], 5):19.1f} Mbps | {np.percentile(rates[:, k], 95):13.1f} Mbps")

# ---------------- a back-of-the-envelope estimate you can do on paper ----------------
# With equal power on every beam, the power cancels and
#     SINR is about  (16 x beta of the serving AP) / (sum of beta of every "leak")
# where a leak is any other user's beam, seen through the channel of user k.
print("\nBACK-OF-THE-ENVELOPE ESTIMATE  (SINR ~ 16 x beta_serving / sum of the leaks' beta, noise ignored)")
print(f"{'user':>5} | {'estimated SINR':>15} | {'estimated speed':>16} | {'simulated average':>18} | {'difference':>10}")
for k in range(n_ue):
    leaks = sum(beta[serving_ap[j], k] for j in range(n_ue) if j != k)
    est_sinr = config.N_ANTENNAS * beta[serving_ap[k], k] / leaks
    est_rate = config.B_HZ * np.log2(1 + est_sinr) / 1e6
    diff = (est_rate - rates[:, k].mean()) / rates[:, k].mean() * 100
    print(f"{k:5d} | {10*np.log10(est_sinr):12.1f} dB | {est_rate:11.0f} Mbps | {rates[:, k].mean():13.0f} Mbps | {diff:+9.0f} %")