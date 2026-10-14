# MOBILITY ON TOP OF THE STATIC PAPER NETWORK (plain Rayleigh channel, Section III-C of the paper).
#
#   STATIC network : users stand still at their starting points.        (what we had before)
#   WALKING network: the SAME users walk (Random Waypoint, 3-60 km/h, one snapshot per second).
#
# The two runs share EVERY random number (same starting points, same shadowing seeds, same fast fading at every second),
# so the only difference between them is that the users walk. Whatever you see in the comparison is caused by walking.
# What changes while a user walks:
#   - the distance to every AP changes             -> the path loss (large-scale gain beta) changes,
#   - the shadowing changes slowly with the walk   -> Eq2Channel.step_state (memory of 50 m, our assumption),
#   - the fast fading g is drawn again every second (at 3.5 GHz and 3+ km/h the channel is uncorrelated after ~1 s).
# Nothing in the other files is changed; this script only imports paper_mobility, paper_models, paper_episodes ...
#
# Saves   img/walk_map.png         one map of the whole network: 121 APs, 100 starting points, the walks of the chosen users
#         img/walk_speed_signal.png  speed and signal strength over time, static vs walking, chosen users
#         img/walk_serving.png       which APs serve each chosen user, second by second
#         img/walk_correlation.png   how similar the walking network stays to the static one
#         tables/walk_ue_summary.csv, tables/walk_checks.csv

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # the files of THIS folder always win
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import paper_config as cfg
import paper_layout as layout
import paper_mobility as mob
import paper_models as pm
import paper_episodes as ep
import paper_simulate as sim
import paper_paths as paths

ap_args = argparse.ArgumentParser()
ap_args.add_argument("--seed", type=int, default=cfg.SEED_DROPS, help="seed of the walk, the channel and the choice of users")
ap_args.add_argument("--n-ue", type=int, default=6, help="how many randomly chosen walking users to show")
ap_args.add_argument("--steps", type=int, default=cfg.EPISODE_STEPS, help="snapshots per walk (1 second each)")
ap_args.add_argument("--show", action="store_true", help="open the windows as well as saving the pictures")
args = ap_args.parse_args()
if not args.show:
    matplotlib.use("Agg")

K, M = cfg.N_UE, cfg.N_AP
T, DT = args.steps, cfg.STEP_S
KMH = 3.6
CLUSTER = f"clusters (L={cfg.L_PER_AP})"
NEAREST = "nearest AP only"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C564B", "#7F7F7F"]
STATIC_COLOR = "#8a8a8a"
records = []


def check(code, text, ok, detail=""):
    ok = bool(ok)
    records.append([code, text, "PASS" if ok else "FAIL", detail])
    print(f"[{code}] {text}{' (' + detail + ')' if detail else ''} -> {'PASS' if ok else 'FAIL'}")


def dbm(p_watt):
    return 10 * np.log10(np.maximum(p_watt, 1e-30) * 1000.0)


# ------------------------------------------------------------------ 1. the walk and the two runs
ap_pos = layout.generate_ap_positions()
rules = {k: v for k, v in sim.standard_rules().items() if k in (NEAREST, CLUSTER)}
channel = pm.Eq2Channel(ap_pos)                       # the channel of the paper: Eq. (1) and (2), plain Rayleigh

pos, vel, speed, leg = mob.random_waypoint(K, T, np.random.default_rng([args.seed, 0]))   # (T, K, 2), steady-state start
frozen = np.repeat(pos[:1], T, axis=0)                # the static network: everybody stays at the starting point


def run(trajectory):
    """Every snapshot of one run. The random numbers depend only on (seed, t), so the static and the walking run
    get the same shadowing innovations and the same fast fading g at every second."""
    state_rng = np.random.default_rng([args.seed, 1])
    state = channel.new_state(trajectory[0], state_rng)
    rec = {"rate": {n: np.zeros((T, K)) for n in rules}, "signal_dbm": {n: np.zeros((T, K)) for n in rules},
           "mask": {n: np.zeros((T, M, K), dtype=bool) for n in rules},
           "strongest": np.zeros((T, K), dtype=int), "best_beta_db": np.zeros((T, K))}
    for t in range(T):
        if t > 0:
            moved = np.linalg.norm(trajectory[t] - trajectory[t - 1], axis=1)
            state = channel.step_state(state, moved, state_rng)
        snap = channel.snapshot(trajectory[t], state, np.random.default_rng([args.seed, 2, t]))
        rec["strongest"][t] = np.argmax(snap["beta"], axis=0)
        rec["best_beta_db"][t] = 10 * np.log10(snap["beta"].max(axis=0))
        for name, r in ep.evaluate(snap, rules).items():
            rec["rate"][name][t], rec["mask"][name][t] = r["rate"], r["mask"]
            rec["signal_dbm"][name][t] = dbm(r["signal"])
    return rec


print(f"Running the walking network and the static network ({T} snapshots, {K} users, {M} APs) ...", flush=True)
walk, stat = run(pos), run(frozen)

# ------------------------------------------------------------------ 2. the randomly chosen users
chosen = np.sort(np.random.default_rng([args.seed, 3]).choice(K, size=args.n_ue, replace=False))
col = {k: COLORS[i % len(COLORS)] for i, k in enumerate(chosen)}
label = {k: i + 1 for i, k in enumerate(chosen)}      # the number printed on the map and in the titles


def serving_set(rec, k, t):
    return np.where(rec["mask"][CLUSTER][t, :, k])[0]


def ue_summary(k):
    path = np.linalg.norm(np.diff(pos[:, k], axis=0), axis=1).sum()
    m = walk["mask"][CLUSTER][:, :, k]
    changes = (m[1:] != m[:-1]).sum(axis=1)                              # AP joins + leaves at every second
    handovers = (np.diff(walk["strongest"][:, k]) != 0)
    r_w, r_s = walk["rate"][CLUSTER][:, k], stat["rate"][CLUSTER][:, k]
    s_w, s_s = walk["signal_dbm"][CLUSTER][:, k], stat["signal_dbm"][CLUSTER][:, k]
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"label": label[k], "ue": int(k), "walked_m": path, "avg_speed_kmh": path / ((T - 1) * DT) * KMH,
            "start_x": pos[0, k, 0], "start_y": pos[0, k, 1], "end_x": pos[-1, k, 0], "end_y": pos[-1, k, 1],
            "static_mean_Mbps": r_s.mean(), "walking_mean_Mbps": r_w.mean(),
            "static_std_Mbps": r_s.std(), "walking_std_Mbps": r_w.std(),
            "static_signal_dBm": s_s.mean(), "walking_signal_dBm": s_w.mean(),
            "corr_speed": corr(r_w, r_s), "corr_signal": corr(s_w, s_s),
            "signal_end_static_dBm": s_s[-1], "signal_end_walking_dBm": s_w[-1],
            "mean_abs_signal_diff_dB": float(np.abs(s_w - s_s).mean()),
            "mean_abs_speed_diff_pct": float(100 * (np.abs(r_w - r_s) / r_s).mean()),
            "aps_ever_serving": int(m.any(axis=0).sum()), "seconds_with_set_change": int((changes > 0).sum()),
            "first_change_s": int(np.argmax(changes > 0) + 1) if (changes > 0).any() else -1,
            "strongest_ap_handovers": int(handovers.sum()),
            "static_set_size": int(stat["mask"][CLUSTER][0, :, k].sum())}


summary = [ue_summary(k) for k in chosen]

# ------------------------------------------------------------------ 3. checks
step = np.linalg.norm(np.diff(pos, axis=0), axis=2)
check("W1", "t = 0 is identical in the static and the walking network (same positions, same channel)",
      np.allclose(walk["rate"][CLUSTER][0], stat["rate"][CLUSTER][0]) and np.allclose(pos[0], frozen[0]))
check("W2", "in the static network the serving APs never change",
      all((stat["mask"][n] == stat["mask"][n][0]).all() for n in rules))
check("W3", "nobody walks faster than 60 km/h (largest step in one second)", step.max() <= cfg.RWP_V_MAX_MS * DT + 1e-6,
      f"{step.max() * KMH / DT:.1f} km/h")
check("W4", "nobody leaves the 1 km x 1 km area", pos.min() >= -1e-9 and pos.max() <= cfg.AREA_M + 1e-9)
check("W5", "every user walks at about the slowest allowed speed or faster on average (3 km/h; a turn at a waypoint shortens one step)",
      (step.sum(axis=0) / ((T - 1) * DT) * KMH).min() >= 2.9,
      f"slowest user {(step.sum(axis=0) / ((T - 1) * DT) * KMH).min():.1f} km/h")
check("W6", "every speed is finite and positive in both networks",
      all(np.isfinite(r["rate"][n]).all() and (r["rate"][n] > 0).all() for r in (walk, stat) for n in rules))
check("W7", "every user is served by at least one AP at every second (cluster rule, walking)",
      bool(walk["mask"][CLUSTER].any(axis=1).all()))
check("W8", "the users on the map are the ones used in the plots", all(pos[0, k, 0] == summary[i]["start_x"] for i, k in enumerate(chosen)))

# ------------------------------------------------------------------ 4. FIGURE 1: the whole network
fig, ax = plt.subplots(figsize=(9.5, 9.5))
ax.set_xlim(-20, cfg.AREA_M + 20)
ax.set_ylim(-20, cfg.AREA_M + 20)
ax.set_aspect("equal")
ax.grid(alpha=0.2)
ax.add_patch(plt.Rectangle((0, 0), cfg.AREA_M, cfg.AREA_M, fill=False, ec="#444", lw=1))
ax.scatter(ap_pos[:, 0], ap_pos[:, 1], marker="^", s=38, c="#c9c9c9", ec="#666", lw=0.6, zorder=2, label=f"AP ({M})")
others = np.setdiff1d(np.arange(K), chosen)
ax.scatter(pos[0, others, 0], pos[0, others, 1], s=12, c="#555", zorder=3, label=f"starting point of a user ({K - len(chosen)} others)")
shown_aps = set()
for k in chosen:
    shown_aps |= set(np.where(walk["mask"][CLUSTER][:, :, k].any(axis=0))[0])
for m in shown_aps:                                                           # APs that serve a chosen user at some moment
    ax.scatter(*ap_pos[m], marker="^", s=38, c="#ffd966", ec="#666", lw=0.6, zorder=2.5)
    ax.annotate(str(m), ap_pos[m], xytext=(3, 4), textcoords="offset points", fontsize=6.5, color="#444", zorder=6)
for k in chosen:
    c = col[k]
    ax.plot(pos[:, k, 0], pos[:, k, 1], "-", color=c, lw=2.2, zorder=4)
    ax.scatter(*pos[0, k], s=150, c=c, ec="k", lw=1.0, zorder=5)
    ax.annotate(str(label[k]), pos[0, k], ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=6)
    ax.scatter(*pos[-1, k], marker="s", s=55, c=c, ec="k", lw=1.0, zorder=5)
    ax.annotate("", xy=pos[-1, k], xytext=pos[max(T - 6, 0), k], arrowprops=dict(arrowstyle="->", color=c, lw=1.8), zorder=5)
ax.scatter([], [], s=150, c="#999", ec="k", label=f"chosen user: start (number) and walk ({T} snapshots, one per second)")
ax.scatter([], [], marker="s", s=55, c="#999", ec="k", label="chosen user: position at the end of the walk")
ax.scatter([], [], marker="^", s=38, c="#ffd966", ec="#666", label="AP that serves a chosen user at some moment (its number)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"The full network: {M} APs, {K} users at their starting points, {len(chosen)} randomly chosen users walking\n"
             f"(Random Waypoint 3-60 km/h, plain Rayleigh channel, exponent {cfg.PAPER_LIKE['alpha']}, {cfg.PAPER_LIKE['shadow_sigma_db']:.0f} dB shadowing)",
             fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(paths.image("walk_map.png"), dpi=120, bbox_inches="tight")

# ------------------------------------------------------------------ 5. FIGURE 2: speed and signal strength, static vs walking
t_axis = np.arange(T) * DT
n = len(chosen)
fig, axs = plt.subplots(n, 2, figsize=(12.5, 2.35 * n), sharex=True, squeeze=False)
for i, k in enumerate(chosen):
    s = summary[i]
    hand = np.where(np.diff(walk["strongest"][:, k]) != 0)[0] + 1
    for j, (key, ylabel) in enumerate((("rate", "speed (Mbps)"), ("signal_dbm", "signal strength (dBm)"))):
        a = axs[i, j]
        a.plot(t_axis, stat[key][CLUSTER][:, k], color=STATIC_COLOR, lw=1.6, label="static (stands at the start)")
        a.plot(t_axis, walk[key][CLUSTER][:, k], color=col[k], lw=1.9, label="walking")
        for h in hand:
            a.axvline(h * DT, color="#b00020", ls=":", lw=1.1, label="strongest AP changes" if h == hand[0] else None)
        a.set_ylabel(ylabel, fontsize=8)
        a.grid(alpha=0.25)
        a.tick_params(labelsize=8)
    axs[i, 0].set_title(f"user {s['label']} (#{s['ue']}):  walks {s['walked_m']:.0f} m at {s['avg_speed_kmh']:.1f} km/h on average;  "
                        f"mean speed {s['static_mean_Mbps']:.0f} (static) vs {s['walking_mean_Mbps']:.0f} (walking) Mbps",
                        fontsize=8.5, loc="left", color=col[k], weight="bold")
    axs[i, 1].set_title(f"signal after {(T - 1) * DT:.0f} s: {s['signal_end_static_dBm']:.0f} dBm (static) vs {s['signal_end_walking_dBm']:.0f} dBm (walking)",
                        fontsize=8.5, loc="left")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=STATIC_COLOR, lw=1.8, label="static (stands at the start)"),
                    Line2D([], [], color="k", lw=2, label="walking (colour of the user)"),
                    Line2D([], [], color="#b00020", ls=":", lw=1.3, label="the strongest AP changes")],
           loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=8.5, frameon=False)
for a in axs[-1]:
    a.set_xlabel("time (s)")
fig.suptitle(f"Speed and signal strength of {n} randomly chosen users: static network vs the same users walking ({CLUSTER}, equal power per AP)",
             fontsize=10.5, y=1.03)
plt.tight_layout()
plt.savefig(paths.image("walk_speed_signal.png"), dpi=115, bbox_inches="tight")

# ------------------------------------------------------------------ 6. FIGURE 3: which APs serve the user, second by second
ncol = 2
nrow = int(np.ceil(n / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow), squeeze=False)
for i, k in enumerate(chosen):
    a = axs[i // ncol, i % ncol]
    m = walk["mask"][CLUSTER][:, :, k]                                # (T, M)
    aps = np.where(m.any(axis=0) | (np.isin(np.arange(M), walk["strongest"][:, k])))[0]
    first = np.array([np.argmax(m[:, ap]) if m[:, ap].any() else T + ap for ap in aps])
    aps = aps[np.argsort(first, kind="stable")]                       # rows: in the order the APs first serve the user
    grid = m[:, aps].T.astype(float)                                  # (rows, T)
    a.pcolormesh(np.arange(T + 1) * DT, np.arange(len(aps) + 1), grid, cmap=matplotlib.colors.ListedColormap(["#f4f4f4", col[k]]),
                 vmin=0, vmax=1, edgecolors="white", linewidth=0.4)
    row_of = {ap: r for r, ap in enumerate(aps)}
    a.scatter(t_axis + DT / 2, [row_of[walk["strongest"][t, k]] + 0.5 for t in range(T)], marker="o", s=14, c="k", zorder=5,
              label="strongest AP (the only one in the conventional rule)")
    a.set_yticks(np.arange(len(aps)) + 0.5)
    a.set_yticklabels([f"AP {ap}" for ap in aps], fontsize=7)
    chg = (m[1:] != m[:-1]).any(axis=1)
    for t in np.where(chg)[0] + 1:
        a.axvline(t * DT, color="#b00020", lw=0.7, alpha=0.5)
    s = summary[i]
    a.set_title(f"user {s['label']} (#{s['ue']}): {s['aps_ever_serving']} different APs serve it during {T * DT:.0f} s, "
                f"set changes in {s['seconds_with_set_change']} s, strongest AP changes {s['strongest_ap_handovers']}x\n"
                f"static: always the same {s['static_set_size']} AP{'s' if s['static_set_size'] != 1 else ''}", fontsize=8, loc="left", color=col[k], weight="bold")
    a.set_xlabel("time (s)", fontsize=8)
    a.tick_params(axis="x", labelsize=8)
for j in range(n, nrow * ncol):
    axs[j // ncol, j % ncol].axis("off")
fig.suptitle(f"Serving APs of the chosen users in the walking network ({CLUSTER}); coloured cell = this AP serves the user; "
             f"thin red lines = the set changes;  black dot = the strongest AP (the only one in the conventional rule)", fontsize=10, y=1.0)
plt.tight_layout()
plt.savefig(paths.image("walk_serving.png"), dpi=115, bbox_inches="tight")

# ------------------------------------------------------------------ 7. FIGURE 4: how similar are the two networks
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
a = axs[0]
top = 0
for i, k in enumerate(chosen):
    x, y = stat["rate"][CLUSTER][:, k], walk["rate"][CLUSTER][:, k]
    a.scatter(x, y, s=20, c=col[k], alpha=0.85, label=f"user {label[k]}:  r = {summary[i]['corr_speed']:.2f}")
    top = max(top, x.max(), y.max())
a.plot([0, top * 1.05], [0, top * 1.05], "k--", lw=1)
a.set_xlabel("speed in the static network (Mbps)")
a.set_ylabel("speed in the walking network (Mbps)")
a.set_title("Same user, same second: static vs walking\n(on the dashed line = walking changed nothing; r = correlation over the 40 s)", fontsize=9)
a.legend(fontsize=8)
a.grid(alpha=0.25)

a = axs[1]
for rule, ls in ((CLUSTER, "-"), (NEAREST, "--")):
    c_speed = [np.corrcoef(stat["rate"][rule][t], walk["rate"][rule][t])[0, 1] for t in range(T)]
    a.plot(t_axis, c_speed, ls, lw=2, label=f"speed, {rule}")
c_sig = [np.corrcoef(stat["signal_dbm"][CLUSTER][t], walk["signal_dbm"][CLUSTER][t])[0, 1] for t in range(T)]
c_beta = [np.corrcoef(stat["best_beta_db"][t], walk["best_beta_db"][t])[0, 1] for t in range(T)]
a.plot(t_axis, c_sig, ":", lw=2, label=f"signal strength, {CLUSTER}")
a.plot(t_axis, c_beta, "-.", lw=1.6, color="#555", label="large-scale gain of the strongest AP")
a.set_ylim(min(0, np.nanmin(c_speed) - 0.05), 1.03)
a.set_xlabel("time since the start of the walk (s)")
a.set_ylabel(f"correlation over all {K} users")
a.set_title(f"How long the walking network stays similar to the static one\n(all {K} users; 1 = identical)", fontsize=9.5)
a.legend(fontsize=7.5)
a.grid(alpha=0.25)

a = axs[2]
xs = np.arange(n)
bars = a.bar(xs, [s["mean_abs_signal_diff_dB"] for s in summary], 0.6, color=[col[k] for k in chosen])
for b, s in zip(bars, summary):
    a.annotate(f"speed {s['mean_abs_speed_diff_pct']:.0f}%", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
a.set_xticks(xs)
a.set_xticklabels([f"user {s['label']}\n{s['walked_m']:.0f} m" for s in summary], fontsize=8)
a.set_ylabel("mean |walking - static| of the signal strength (dB)")
a.set_title("How far each chosen user moves away from its static self\n(bar = mean signal difference in dB; label = mean speed difference in %)", fontsize=9.5)
a.grid(axis="y", alpha=0.25)
a.set_ylim(0, max(s["mean_abs_signal_diff_dB"] for s in summary) * 1.18)
fig.suptitle("Static network vs walking network (the same random numbers in both; only the walking differs)", fontsize=10.5)
plt.tight_layout()
plt.savefig(paths.image("walk_correlation.png"), dpi=115, bbox_inches="tight")

# ------------------------------------------------------------------ 8. tables and the printed summary
header = list(summary[0].keys())
with open(paths.table("walk_ue_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows([[f"{v:.5g}" if isinstance(v, float) else v for v in s.values()] for s in summary])
with open(paths.table("walk_checks.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["code", "check", "result", "detail"])
    w.writerows(records)

print(f"\nChosen users (seed {args.seed}):")
print(f"{'no':>3} {'user':>5} {'walked m':>9} {'km/h':>6} {'static Mbps':>12} {'walking Mbps':>13} {'r speed':>8} {'r signal':>9} "
      f"{'APs used':>9} {'set changes':>12} {'handovers':>10}")
for s in summary:
    print(f"{s['label']:>3} {s['ue']:>5} {s['walked_m']:9.0f} {s['avg_speed_kmh']:6.1f} {s['static_mean_Mbps']:12.0f} "
          f"{s['walking_mean_Mbps']:13.0f} {s['corr_speed']:8.2f} {s['corr_signal']:9.2f} {s['aps_ever_serving']:9d} "
          f"{s['seconds_with_set_change']:12d} {s['strongest_ap_handovers']:10d}")
print(f"\nAll {K} users: mean speed static {stat['rate'][CLUSTER].mean():.0f} Mbps, walking {walk['rate'][CLUSTER].mean():.0f} Mbps; "
      f"correlation of the speeds at the last second {c_speed[-1]:.2f} (clusters)")
print("Checks passed:", sum(r[2] == "PASS" for r in records), "of", len(records))
print("Saved img/walk_map.png, walk_speed_signal.png, walk_serving.png, walk_correlation.png and tables/walk_ue_summary.csv, walk_checks.csv")
if args.show:
    plt.show()