# Builds the channel for our 2 APs x 4 users x 16 antennas and checks it.

import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from path_loss_model import beta_from_positions
from channel_model import rayleigh_fading, full_channel

np.set_printoptions(linewidth=150)

_, _, beta = beta_from_positions(generate_ap_positions(), generate_ue_positions())

rng = np.random.default_rng(config.SEED)
H = full_channel(beta, rng)

print("Shape of H (APs, users, antennas):", H.shape)
print("\nThe 16 channel numbers between AP0 and User0 (first 4 shown):")
print(H[0, 0, :4])
print("their sizes |h|:", np.abs(H[0, 0, :4]))

# Total strength of each AP-user channel: add up |h|^2 over the 16 antennas
strength = np.sum(np.abs(H) ** 2, axis=2)             # shape (2, 4)
expected = config.N_ANTENNAS * beta                    # what we expect on average

print("\nStrength of each channel in THIS random draw (rows = APs, columns = users):")
print(strength)
print("\nStrength we expect on average (16 x beta):")
print(expected)
print("\nRatio draw / expected (1.0 would be exactly average):")
print((strength / expected).round(2))

#np.savetxt("step3_strength_ratio.csv", strength / expected, delimiter=",", fmt="%.4f",
#           header="user0,user1,user2,user3")
#print("\nSaved step3_strength_ratio.csv")

# ---------------- CHECKS ----------------
print("\nCHECKS")

# 1. Right shape: 2 APs x 4 users x 16 antennas
ok1 = H.shape == (config.N_AP, config.N_UE, config.N_ANTENNAS)
print(f"[1] H has shape {H.shape}, expected ({config.N_AP}, {config.N_UE}, {config.N_ANTENNAS}) -> {'PASS' if ok1 else 'FAIL'}")

# 2. Same seed gives the same numbers, a different seed gives different ones
H_same = full_channel(beta, np.random.default_rng(config.SEED))
H_other = full_channel(beta, np.random.default_rng(config.SEED + 1))
ok2 = bool(np.array_equal(H, H_same)) and not np.allclose(H, H_other)
print(f"[2] same seed -> identical numbers, different seed -> different numbers -> {'PASS' if ok2 else 'FAIL'}")

# 3. The random numbers g must have average 0 and average power 1
g = rayleigh_fading(2000, config.N_UE, config.N_ANTENNAS, np.random.default_rng(10))
mean_g = abs(g.mean())
power_g = np.mean(np.abs(g) ** 2)
var_re, var_im = g.real.var(), g.imag.var()
ok3 = mean_g < 0.01 and abs(power_g - 1) < 0.02 and abs(var_re - 0.5) < 0.01 and abs(var_im - 0.5) < 0.01
print(f"[3] random numbers: average {mean_g:.4f} (expected 0), power {power_g:.4f} (expected 1), "
      f"real/imag variance {var_re:.3f}/{var_im:.3f} (expected 0.5 each) -> {'PASS' if ok3 else 'FAIL'}")

# 4. Averaged over many random draws, the strength of every channel must be 16 x beta
trials = 5000
rng4 = np.random.default_rng(20)
avg_strength = np.zeros_like(beta)
for _ in range(trials):
    avg_strength += np.sum(np.abs(full_channel(beta, rng4)) ** 2, axis=2)
avg_strength /= trials
rel_err = np.abs(avg_strength - expected) / expected
ok4 = rel_err.max() < 0.03
print(f"[4] average strength over {trials} draws vs 16 x beta: largest error {rel_err.max()*100:.2f}% -> {'PASS' if ok4 else 'FAIL'}")

# 5. Channel hardening: the relative randomness of the strength is 1/sqrt(N)
print("[5] channel hardening (relative randomness of the strength, theory 1/sqrt(N)):")
ok5 = True
rng5 = np.random.default_rng(30)
for n in [1, 4, 16, 64, 256]:
    gg = rayleigh_fading(4000, 1, n, rng5)
    s = np.sum(np.abs(gg) ** 2, axis=2).ravel()
    rel = s.std() / s.mean()
    theory = 1 / np.sqrt(n)
    good = abs(rel - theory) / theory < 0.1
    ok5 &= bool(good)
    print(f"      N={n:4d}: measured {rel:.3f}, theory {theory:.3f} -> {'OK' if good else 'OFF'}")
print(f"    -> {'PASS' if ok5 else 'FAIL'}")