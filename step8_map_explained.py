# A closer look at the RIGHT picture of step8a_map.png (the cluster rule): who serves whom,
# how far the serving APs are, and how much each of them adds for TWO chosen users:
#   user A = a user that stands close to one AP
#   user B = a user that stands between several APs
# Same drop as in step8a_map.png (random seed 3). 
# (the two users are chosen automatically)

import sys
import numpy as np
import matplotlib.pyplot as plt

import mid_config as cfg
import mid_layout
import mid_simulate as sim
from precoding import signal_seen_by_users, mrt_precoders

ap_pos = mid_layout.generate_ap_positions()
RULES = sim.standard_rules()
CLUSTER = [name for name in RULES if "cluster" in name][0]
drop = sim.simulate_drop(ap_pos, np.random.default_rng(3), RULES)
ue, dist, H = drop["ue"], drop["dist"], drop["H"]
r = drop["rules"][CLUSTER]
mask, power = r["mask"], r["power"]
n_aps = mask.sum(axis=0)

# the wanted signal that every serving AP contributes to its user: A[m, k, k] = h_mk^H w_mk
A = signal_seen_by_users(H, mrt_precoders(H, power))
amp = np.abs(np.array([A[:, k, k] for k in range(cfg.N_UE)]).T)          # (APs, users)


def dbm(x):
    return 10 * np.log10(x) + 30


# how much of the final signal does the BEST serving AP give alone? (1 = it gives everything)
best_share = np.array([amp[mask[:, k], k].max() ** 2 / r["signal"][k] for k in range(cfg.N_UE)])

# ---------------- table 1: every user ----------------
print("WHICH APs SERVE EACH USER (cluster rule, L = 3, the drop of step8a_map.png)")
print(f"{'user':>4} {'APs':>4} | {'serving APs, nearest first (distance in m)':<72} | {'best AP alone gives':>20}")
for k in range(cfg.N_UE):
    serving = np.where(mask[:, k])[0]
    order = serving[np.argsort(dist[serving, k])]
    text = ", ".join(f"AP{m} ({dist[m, k]:.0f})" for m in order)
    print(f"{k:4d} {n_aps[k]:4d} | {text:<72} | {best_share[k] * 100:17.0f} % of the final signal")
print(f"\nnumber of APs per user: smallest {n_aps.min()}, largest {n_aps.max()}, average {n_aps.mean():.2f}")

# ---------------- table 2: every AP ----------------
print("\nWHICH USERS EACH AP SERVES (nearest first)")
for m in range(cfg.N_AP):
    users = np.where(mask[m])[0]
    order = users[np.argsort(dist[m, users])]
    print(f"   AP{m:<3d} | " + ", ".join(f"user{k} ({dist[m, k]:.0f} m)" for k in order))

# ---------------- the two chosen users ----------------
multi = np.where(n_aps >= 2)[0]
if "--users" in sys.argv:
    i = sys.argv.index("--users")
    user_a, user_b = int(sys.argv[i + 1]), int(sys.argv[i + 2])
else:
    user_a = int(multi[np.argmax(best_share[multi])])       # one AP gives almost everything
    user_b = int(multi[np.argmin(best_share[multi])])       # no AP dominates
details = {}
for tag, user in (("A", user_a), ("B", user_b)):
    serving = np.where(mask[:, user])[0]
    serving = serving[np.argsort(dist[serving, user])]
    best = amp[serving, user].max() ** 2
    details[tag] = (user, serving)
    print(f"\nUSER {user} (user {tag}): {len(serving)} serving APs, speed {r['rate'][user]:.0f} Mbps, SINR {10 * np.log10(r['sinr'][user]):.1f} dB")
    for m in serving:
        print(f"   AP{m:<3d} {dist[m, user]:6.1f} m away: its beam alone gives {dbm(amp[m, user] ** 2):6.1f} dBm "
              f"({10 * np.log10(amp[m, user] ** 2 / best):+6.1f} dB compared with the best AP)")
    print(f"   all together (the amplitudes add up): {dbm(r['signal'][user]):.1f} dBm = {10 * np.log10(r['signal'][user] / best):.1f} dB more than the best AP alone")
    print(f"   interference (the beams meant for other users): {dbm(r['interference'][user]):.1f} dBm")

# ---------------- CHECKS ----------------
print("\nCHECKS")
ok1 = bool(np.all(mask.sum(axis=1) == cfg.L_PER_AP))
print(f"[1] every AP serves exactly {cfg.L_PER_AP} users (every triangle has {cfg.L_PER_AP} lines) -> {'PASS' if ok1 else 'FAIL'}")
ok2 = int(mask.sum()) == cfg.N_AP * cfg.L_PER_AP == int(n_aps.sum())
print(f"[2] lines in the picture: {int(mask.sum())} = {cfg.N_AP} APs x {cfg.L_PER_AP} = sum of the APs of all users ({int(n_aps.sum())}) -> {'PASS' if ok2 else 'FAIL'}")
ok3 = bool(np.all(mask.any(axis=0)))
print(f"[3] every user has at least one line -> {'PASS' if ok3 else 'FAIL'}")
ok4 = True
for tag, (user, serving) in details.items():
    combined = amp[serving, user].sum() ** 2
    ok4 &= bool(abs(combined - r["signal"][user]) / r["signal"][user] < 1e-9 and r["signal"][user] >= amp[serving, user].max() ** 2)
print(f"[4] wanted signal = (sum of the AP amplitudes)^2, never below the best AP alone (both users) -> {'PASS' if ok4 else 'FAIL'}")
ok5 = abs(n_aps.mean() - cfg.N_AP * cfg.L_PER_AP / cfg.N_UE) < 1e-9
print(f"[5] the average number of APs per user is APs x L / users = {cfg.N_AP * cfg.L_PER_AP / cfg.N_UE:.2f} -> {'PASS' if ok5 else 'FAIL'}")

# ---------------- picture ----------------
fig, ax = plt.subplots(2, 2, figsize=(16, 15), gridspec_kw={"width_ratios": [1.15, 1]})
for row, (tag, (user, serving)) in enumerate(details.items()):
    a = ax[row, 0]
    for m, k in zip(*np.where(mask)):
        highlighted = k == user
        a.plot([ap_pos[m, 0], ue[k, 0]], [ap_pos[m, 1], ue[k, 1]], "-", color="tab:red" if highlighted else "lightgray",
               lw=3 if highlighted else 1, zorder=2 if highlighted else 1)
        if highlighted:
            mid = (ap_pos[m] + ue[k]) / 2
            a.text(mid[0], mid[1], f"{dist[m, k]:.0f} m", color="tab:red", fontsize=10, fontweight="bold",
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1), zorder=6)
    for m in range(cfg.N_AP):
        a.scatter(*ap_pos[m], marker="^", s=170, zorder=3, color="tab:red" if m in serving else "black", edgecolor="black")
        a.text(ap_pos[m, 0] + 8, ap_pos[m, 1] - 18, f"AP{m}", fontsize=8)
    a.scatter(ue[:, 0], ue[:, 1], s=55, color="tab:orange", edgecolor="black", zorder=4)
    a.scatter(*ue[user], s=170, color="tab:orange", edgecolor="red", linewidth=3, zorder=5)
    for k in range(cfg.N_UE):
        a.text(ue[k, 0] + 6, ue[k, 1] + 6, str(k), fontsize=9 if k == user else 7,
               fontweight="bold" if k == user else "normal", zorder=7)
    a.set_xlim(0, cfg.AREA_M)
    a.set_ylim(0, cfg.AREA_M)
    a.set_aspect("equal")
    a.set_xlabel("x (m)")
    a.set_ylabel("y (m)")
    kind = "stands close to one AP" if tag == "A" else "stands between several APs"
    a.set_title(f"User {user} ({kind}): {len(serving)} serving APs (red triangles)\n"
                f"red lines show the distances, gray lines are the beams of all the other users", fontsize=10)
    a.grid(True, alpha=0.2)

    b = ax[row, 1]
    each = [dbm(amp[m, user] ** 2) for m in serving]
    labels = [f"AP{m}\n{dist[m, user]:.0f} m" for m in serving] + ["all\ntogether"]
    values = each + [dbm(r["signal"][user])]
    floor = min(values + [dbm(r["interference"][user])]) - 8
    colours = ["tab:red"] * len(serving) + ["tab:blue"]
    b.vlines(range(len(values)), floor, values, color=colours, lw=3)
    b.scatter(range(len(values)), values, s=140, color=colours, zorder=3)
    for i, v in enumerate(values):
        b.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=10)
    b.axhline(dbm(r["interference"][user]), color="black", ls="--", lw=2, label="interference at this user")
    b.set_xticks(range(len(values)))
    b.set_xticklabels(labels, fontsize=9)
    b.set_ylabel("power at the user (dBm)")
    b.set_ylim(floor, max(values) + 6)
    b.grid(True, axis="y", alpha=0.3)
    gain = 10 * np.log10(r["signal"][user] / amp[serving, user].max() ** 2)
    b.set_title(f"User {user}: what each serving AP contributes\n"
                f"together {gain:.1f} dB more than the best AP alone; speed {r['rate'][user]:.0f} Mbps, "
                f"SINR {10 * np.log10(r['sinr'][user]):.1f} dB", fontsize=10)
    b.legend(loc="lower right")
plt.tight_layout()
plt.savefig("img/step8a_map_explained.png", dpi=110)
print("\nSaved step8a_map_explained.png")
if "--no-show" not in sys.argv:
    plt.show()