# Aims the antennas (MRT), measures what the users receive, and checks it.

import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import full_channel
from precoding import mrt_precoders, no_aim_precoders, signal_seen_by_users

np.set_printoptions(linewidth=150, suppress=True)

_, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())
n_ap, n_ue, n_ant = config.N_AP, config.N_UE, config.N_ANTENNAS
idx = np.arange(n_ue)

# The same random channel as in Step 3 (seed 0)
H = full_channel(beta, np.random.default_rng(config.SEED))

# Every link gets 1 unit of power, so that we see only the effect of aiming
p = np.ones((n_ap, n_ue))
W_aim = mrt_precoders(H, p)
W_off = no_aim_precoders(H, p)
A_aim = signal_seen_by_users(H, W_aim)
A_off = signal_seen_by_users(H, W_off)

# What ONE antenna would give on average = power x beta. We use it as the yardstick.
reference = p * beta
gain_aim = np.abs(A_aim[:, idx, idx]) ** 2 / reference      # (2, 4)
gain_off = np.abs(A_off[:, idx, idx]) ** 2 / reference

print("Received power compared with ONE antenna (1.0 = what one antenna gives on average)")
print("\nAIMED at the user (rows = APs, columns = users):")
print(gain_aim.round(1))
print("\nNOT aimed (all antennas send the same thing):")
print(gain_off.round(2))
print(f"\naverage over these 8 links: aimed {gain_aim.mean():.1f} x, not aimed {gain_off.mean():.2f} x")
print("(this is just ONE random draw, so 'not aimed' can be lucky; the 5000-draw checks below give the true averages)")

# Leakage table: AP0 only. Row = the user who listens, column = the user the beam is aimed at.
leak = np.abs(A_aim[0]) ** 2 / beta[0][:, None]
print("\nAP0 sends 4 beams, one per user. Row k = what user k receives, column j = beam aimed at user j")
print("(diagonal = the wanted signal, off-diagonal = leakage, all compared with one antenna):")
print(leak.round(1))

#np.savetxt("step4_gain_aimed.csv", gain_aim, delimiter=",", fmt="%.3f", header="user0,user1,user2,user3")
#np.savetxt("step4_gain_not_aimed.csv", gain_off, delimiter=",", fmt="%.3f", header="user0,user1,user2,user3")
#print("\nSaved step4_gain_aimed.csv and step4_gain_not_aimed.csv")

# ---------------- CHECKS ----------------
print("\nCHECKS")

# 1. The weights use exactly the power we gave (squared length = p)
err = np.max(np.abs(np.linalg.norm(W_aim, axis=2) ** 2 - p))
ok1 = err < 1e-9
print(f"[1] weights use exactly the power given: max error {err:.1e} -> {'PASS' if ok1 else 'FAIL'}")

# 2. With aiming, the wanted power must equal p x ||h||^2 (the strength of the channel)
strength = np.sum(np.abs(H) ** 2, axis=2)
ok2 = bool(np.allclose(np.abs(A_aim[:, idx, idx]) ** 2, p * strength))
print(f"[2] aimed power equals the full channel strength (all 16 antennas used) -> {'PASS' if ok2 else 'FAIL'}")

# 3. With aiming, all 16 "arrows" point the same way: the total is a real, positive number
wanted = A_aim[:, idx, idx]
ok3 = bool(np.all(wanted.real > 0) and np.max(np.abs(wanted.imag) / np.abs(wanted)) < 1e-9)
print(f"[3] with aiming the arrows line up (total has no leftover twist) -> {'PASS' if ok3 else 'FAIL'}")

# Many random draws for the next checks
trials = 5000
rng = np.random.default_rng(100)
g_aim = np.zeros(trials)
g_off = np.zeros(trials)
g_leak = np.zeros(trials)
never_worse = True
for t in range(trials):
    Ht = full_channel(beta, rng)
    Wa, Wo = mrt_precoders(Ht, p), no_aim_precoders(Ht, p)
    Aa, Ao = signal_seen_by_users(Ht, Wa), signal_seen_by_users(Ht, Wo)
    ga = np.abs(Aa[:, idx, idx]) ** 2 / reference
    go = np.abs(Ao[:, idx, idx]) ** 2 / reference
    never_worse &= bool(np.all(ga >= go - 1e-12))
    g_aim[t], g_off[t] = ga.mean(), go.mean()
    off_diag = ~np.eye(n_ue, dtype=bool)
    g_leak[t] = np.mean([(np.abs(Aa[m]) ** 2 / beta[m][:, None])[off_diag].mean() for m in range(n_ap)])

# 4. On average aiming gives N times more power than one antenna
ok4 = abs(g_aim.mean() - n_ant) / n_ant < 0.03
print(f"[4] average gain when aimed: {g_aim.mean():.2f} (expected {n_ant}) -> {'PASS' if ok4 else 'FAIL'}")

# 5. Without aiming there is no gain on average
ok5 = abs(g_off.mean() - 1) < 0.03
print(f"[5] average gain when NOT aimed: {g_off.mean():.3f} (expected 1) -> {'PASS' if ok5 else 'FAIL'}")

# 6. Other users only get a little leakage (like one antenna), not the big gain
ok6 = abs(g_leak.mean() - 1) < 0.03
print(f"[6] average leakage to the other users: {g_leak.mean():.3f} (expected 1, i.e. {n_ant} times less than the target) -> {'PASS' if ok6 else 'FAIL'}")

# 7. Aiming is never worse than not aiming, in any of the draws
print(f"[7] in all {trials} draws aimed is at least as strong as not aimed -> {'PASS' if never_worse else 'FAIL'}")