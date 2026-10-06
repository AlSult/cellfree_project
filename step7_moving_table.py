# The users walk for 40 seconds. At every second the distances, the channel and
# the speeds are recomputed, for BOTH rules (Step 5: nearest AP, Step 6: cell-free).

import numpy as np

import config
from ap_location_generation import generate_ap_positions
from ue_location_generation import generate_ue_positions
from mobility_model import generate_trajectories
from path_loss_model import beta_from_positions
from channel_model import full_channel
from power_allocation import nearest_ap_mask, cooperative_mask, equal_power_allocation
from precoding import mrt_precoders
from rate_calculation import sinr_and_rate
from closed_form_rate import expected_rate

np.set_printoptions(linewidth=150, suppress=True)
n_ap, n_ue, T = config.N_AP, config.N_UE, config.N_STEPS

ap_pos, ue_start = generate_ap_positions(), generate_ue_positions()
pos, speed = generate_trajectories(ue_start)                 # (T, users, 2)

# ---------------- what the walk looks like ----------------
print("WALKING SPEEDS (m/s):", speed.round(2))
print(f"{'user':>5} | {'starts at':>14} | {'ends at':>16} | {'distance walked':>16}")
for k in range(n_ue):
    walked = np.sum(np.linalg.norm(np.diff(pos[:, k], axis=0), axis=1))
    print(f"{k:5d} | ({pos[0, k, 0]:5.1f},{pos[0, k, 1]:6.1f}) | ({pos[-1, k, 0]:6.1f},{pos[-1, k, 1]:6.1f}) | {walked:13.1f} m")

# ---------------- distances, channel gains and serving rules at EVERY time step ----------------
dist_t = np.zeros((T, n_ap, n_ue))
beta_t = np.zeros((T, n_ap, n_ue))
for t in range(T):
    dist_t[t], _, beta_t[t] = beta_from_positions(ap_pos, pos[t])
masks_t = {"nearest": np.array([nearest_ap_mask(dist_t[t]) for t in range(T)]),   # the nearest AP can CHANGE while walking
           "cellfree": np.array([cooperative_mask() for _ in range(T)])}
power_t = {name: np.array([equal_power_allocation(masks_t[name][t]) for t in range(T)]) for name in masks_t}

serving = np.argmax(masks_t["nearest"], axis=1)                      # (T, users): which AP serves each user
changes = (np.diff(serving, axis=0) != 0).sum(axis=0)
print(f"\nSTEP 5 RULE: number of times the serving AP changes during the walk (a 'handover'): {changes}")
print("CELL-FREE RULE: no handover is ever needed, every AP always serves every user")

# ---------------- ONE run (one random channel per second), and an ensemble average ----------------
trials = 300
rng_single = np.random.default_rng(config.SEED)
rng_ens = np.random.default_rng(1000)
single = {name: np.zeros((T, n_ue)) for name in masks_t}
ens = {name: np.zeros((trials, T, n_ue)) for name in masks_t}
for t in range(T):
    Hs = full_channel(beta_t[t], rng_single)
    for name in masks_t:
        single[name][t] = sinr_and_rate(Hs, mrt_precoders(Hs, power_t[name][t]))[3]
    for r in range(trials):
        Hr = full_channel(beta_t[t], rng_ens)
        for name in masks_t:
            ens[name][r, t] = sinr_and_rate(Hr, mrt_precoders(Hr, power_t[name][t]))[3]
mean = {name: ens[name].mean(axis=0) for name in masks_t}
ref = {name: np.array([expected_rate(beta_t[t], power_t[name][t]) for t in range(T)]) for name in masks_t}

label = {"nearest": "Step 5 (nearest AP)", "cellfree": "Step 6 (cell-free)"}
for name in masks_t:
    print(f"\n{label[name].upper()}: speed of every user while walking (Mbps)")
    print(f"{'user':>5} | {'time average':>13} {'lowest':>8} {'highest':>8} | {'one run: flicker around the average':>38}")
    for k in range(n_ue):
        flicker = np.std(single[name][:, k] - mean[name][:, k]) / np.mean(mean[name][:, k]) * 100
        print(f"{k:5d} | {mean[name][:, k].mean():13.0f} {mean[name][:, k].min():8.0f} {mean[name][:, k].max():8.0f} | "
              f"{flicker:33.0f} % of the average")
    print(f"  sum of the 4 time-averages: {mean[name].mean(axis=0).sum():.0f} Mbps")

# ---------------- CHECKS ----------------
print("\nCHECKS")
step_disp = np.linalg.norm(np.diff(pos, axis=0), axis=2) / config.DT_S
ok1 = bool(np.all(step_disp <= speed[None, :] * (1 + 1e-9)))
print(f"[1] nobody moves faster than its own speed, in any step -> {'PASS' if ok1 else 'FAIL'}")
exact = np.abs(step_disp - speed[None, :]) / speed[None, :] < 1e-6
ok2 = exact.mean() > 0.9
print(f"[2] {exact.mean()*100:.1f}% of the steps are exactly speed x time ({(~exact).sum()} step(s) contain a bounce, where a shorter straight chord is expected) -> {'PASS' if ok2 else 'FAIL'}")
ok3 = bool(np.all((pos >= -1e-9) & (pos <= config.AREA_M + 1e-9)))
print(f"[3] everybody stays inside the {config.AREA_M:.0f} x {config.AREA_M:.0f} m map at all times -> {'PASS' if ok3 else 'FAIL'}")
ok4 = bool(np.allclose(pos[0], ue_start))
print(f"[4] at time 0 everybody stands where Step 1 put them -> {'PASS' if ok4 else 'FAIL'}")
ok5 = bool(np.all(masks_t["nearest"].sum(axis=1) == 1))
print(f"[5] Step 5 rule: at every moment every user has exactly one serving AP -> {'PASS' if ok5 else 'FAIL'}")
print(f"[6] {trials}-draw average of the simulation vs the paper-and-pencil formula (no random numbers):")
ok6 = True
for name in masks_t:
    err = np.abs(mean[name] - ref[name]) / mean[name]
    ok6 &= bool(err.mean() < 0.15)
    print(f"      {label[name]:20s} average difference {err.mean()*100:4.1f}%, largest {err.max()*100:4.1f}%  "
          f"(formula is {np.mean(ref[name]/mean[name]-1)*100:+.0f}% on average)")
print(f"    -> {'PASS' if ok6 else 'FAIL'}   (a small underestimate is expected: the formula uses averages, the real interference is sometimes much weaker)")

#np.savetxt("step7_average_speeds.csv", np.column_stack([mean["nearest"].mean(axis=0), mean["cellfree"].mean(axis=0)]),
#           delimiter=",", fmt="%.2f", header="step5_nearest_Mbps,step6_cellfree_Mbps")
#print("\nSaved step7_average_speeds.csv")