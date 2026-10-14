# STEP 10 (data generation): builds the MASTER DATASET for the federated learning.
#   Paper-size network (121 APs x 64 antennas, 100 users), plain Rayleigh channel (Section III-C), exponent 3.8,
#   8 dB shadowing with 50 m memory, Random Waypoint 3-60 km/h, 1 snapshot per second, episodes of 40 snapshots.
#   For every snapshot: the INPUTS (what the APs would measure) and the LABEL (the optimized power split, 947 numbers).
#   No train/test split and no per-AP files are made here: the master file only carries the episode number and the time
#   step of every row, so any split by whole episodes can be made later without generating anything again.
#
# What one snapshot contains (units in brackets):
#   pos (K,2) [m]  vel (K,2) [m/s]  beta_db (M,K) [dB, float16]  traffic (K) [all 1.0 = equal demand, see TRAFFIC_DEMAND]
#   cand (M,7)  : the user that sits in every slot of every AP (strongest first); needed to read the label
#   label_pf (947): 121 x 7 power shares of the slots, then 100 scheduling shares   (optimized for 'pf'; 'sum' with --also-sum)
#   rates_pf (K) [Mbps]: the speeds that this label gives (for the audit)       objective_pf: the score of the label
# The full channel H is NOT stored (12 MB per snapshot). It is recreated exactly from the seeds:
#   seeds are [DATASET_SEED, episode, 0] trajectory, [DATASET_SEED, episode, 1] shadowing, [DATASET_SEED, episode, 2, t] fading.
#
# Files:  data/<name>/ep_0000.npz ...  one file per finished episode (also the resume points)
#         data/ppfml_<name>.npz         the master file (all episodes together) + data/ppfml_<name>_meta.json
#         tables/dataset_<name>_summary.csv, tables/dataset_<name>_checks.csv, img/dataset_<name>_overview.png
# Run:    python run_generate_dataset.py                          (100 episodes, 1 worker)
#         python run_generate_dataset.py --episodes 100 --workers 4     (faster on a computer with 4 cores)
#         python run_generate_dataset.py --episodes 2 --name test       (quick test, separate files)
# It can be stopped and started again: finished episodes are skipped.

import argparse
import os
import sys

ap_ = argparse.ArgumentParser()
ap_.add_argument("--episodes", type=int, default=100)
ap_.add_argument("--steps", type=int, default=40, help="snapshots per episode (default 40 = 40 seconds)")
ap_.add_argument("--iters", type=int, default=400, help="optimizer steps per snapshot")
ap_.add_argument("--workers", type=int, default=1, help="episodes computed in parallel")
ap_.add_argument("--name", type=str, default="dataset", help="name of this dataset (separate folder and files)")
ap_.add_argument("--also-sum", action="store_true", help="also store the label of the objective 'sum'")
ap_.add_argument("--no-verify", action="store_true")
args = ap_.parse_args()

if args.workers > 1:                                   # one thread per worker, otherwise they fight for the cores
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"

import json
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

import paper_config as cfg
import paper_layout as layout
import paper_labels as lab
import paper_mobility as mob
import paper_paths as paths
from paper_models import Eq2Channel

DATASET_SEED = 2026
L = 7                                                         # slots per AP (the paper's 7 numbers per AP)
K, M = cfg.N_UE, cfg.N_AP
TRAFFIC_DEMAND = getattr(cfg, "TRAFFIC_DEMAND", 1.0)          # equal demand for all users (a simplification, see the thesis notes)
KINDS = ["pf", "sum"] if args.also_sum else ["pf"]
LABEL_LEN = M * L + K                                          # 947
OUT_DIR = paths.data(args.name)
MASTER = paths.data(f"ppfml_{args.name}.npz")
META = paths.data(f"ppfml_{args.name}_meta.json")


# ------------------------------------------------------------------ one episode
def walk_episode(e, steps, channel):
    """The walking users and their channel, recreated from the seeds only. Yields (t, pos, vel, snapshot)."""
    pos, vel, speed, leg = mob.random_waypoint(K, steps, np.random.default_rng([DATASET_SEED, e, 0]))
    state_rng = np.random.default_rng([DATASET_SEED, e, 1])
    state = channel.new_state(pos[0], state_rng)
    for t in range(steps):
        if t > 0:
            state = channel.step_state(state, np.linalg.norm(pos[t] - pos[t - 1], axis=1), state_rng)
        yield t, pos[t], vel[t], channel.snapshot(pos[t], state, np.random.default_rng([DATASET_SEED, e, 2, t]))


def episode_file(e):
    return os.path.join(OUT_DIR, f"ep_{e:04d}.npz")


def is_finished(e, steps, kinds):
    try:
        with np.load(episode_file(e)) as z:
            return z["pos"].shape[0] == steps and all(f"label_{k}" in z.files for k in kinds)
    except Exception:
        return False


def make_episode(task):
    e, steps, iters, kinds, out_dir = task
    ap_pos = layout.generate_ap_positions()
    channel = Eq2Channel(ap_pos)
    d = {"pos": np.zeros((steps, K, 2), np.float32), "vel": np.zeros((steps, K, 2), np.float32),
         "beta_db": np.zeros((steps, M, K), np.float16), "traffic": np.full((steps, K), TRAFFIC_DEMAND, np.float32),
         "cand": np.zeros((steps, M, L), np.int16)}
    for k in kinds:
        d[f"label_{k}"] = np.zeros((steps, LABEL_LEN), np.float32)
        d[f"rates_{k}"] = np.zeros((steps, K), np.float32)
        d[f"objective_{k}"] = np.zeros(steps, np.float64)
    t0 = time.time()
    for t, pos, vel, snap in walk_episode(e, steps, channel):
        prob = lab.LabelProblem(snap["H"], snap["beta"], L)
        d["pos"][t], d["vel"][t] = pos, vel
        d["beta_db"][t] = 10 * np.log10(snap["beta"])
        d["cand"][t] = prob.cand
        for k in kinds:
            res = lab.optimize(prob, k, iters=iters)                     # the normal start (3 strongest users), see run_optimizer_check.py
            d[f"label_{k}"][t] = lab.label_vector(prob, res["p"])
            d[f"rates_{k}"][t] = prob.rates(res["p"])
            d[f"objective_{k}"][t] = res["objective"]
    path = os.path.join(out_dir, f"ep_{e:04d}.npz")
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)                                                # the file appears only when it is complete
    return e, time.time() - t0


# ------------------------------------------------------------------ merge
def merge(n_episodes, steps, kinds):
    names = ["pos", "vel", "beta_db", "traffic", "cand"] + [f"{a}_{k}" for k in kinds for a in ("label", "rates", "objective")]
    parts = {n: [] for n in names}
    for e in range(n_episodes):
        with np.load(episode_file(e)) as z:
            for n in names:
                parts[n].append(z[n])
    out = {n: np.concatenate(v, axis=0) for n, v in parts.items()}
    out["episode"] = np.repeat(np.arange(n_episodes), steps).astype(np.int32)
    out["t"] = np.tile(np.arange(steps), n_episodes).astype(np.int16)
    out["ap_pos"] = layout.generate_ap_positions().astype(np.float32)
    np.savez_compressed(MASTER, **out)
    meta = {
        "format_version": 1, "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "episodes": n_episodes, "snapshots_per_episode": steps, "snapshots": n_episodes * steps,
        "users": K, "aps": M, "antennas": cfg.N_ANTENNAS, "slots_per_ap": L, "label_length": LABEL_LEN,
        "label_kinds": kinds, "optimizer_steps": args.iters, "optimizer_start": "equal power on the 3 strongest users",
        "dataset_seed": DATASET_SEED, "layout_seed": cfg.SEED_LAYOUT, "ap_jitter_m": cfg.AP_JITTER_M,
        "seeds": "[DATASET_SEED, episode, 0] trajectory; [.., 1] shadowing; [.., 2, t] fast fading",
        "channel": "plain Rayleigh (paper Section III-C, Eq. 1-2), exponent 3.8, shadowing 8 dB with 50 m memory",
        "mobility": "Random Waypoint 3-60 km/h, no pause, 1 s per snapshot",
        "traffic_demand": f"{TRAFFIC_DEMAND} for every user (EQUAL demand: a simplification; the paper has per-user demands)",
        "not_modelled": ["traffic demands (equal)", "IoT sensor readings", "pilot contamination / channel estimation error",
                         "antenna gains (G_m = G_k = 1)", "wrap-around at the border"],
        "label_layout": "label[0:847] = 121 x 7 power shares (slot s of AP m = cand[m, s], strongest first, sum over s <= 1); "
                        "label[847:947] = share of the network power spent on each user (sums to 1)",
        "units": {"pos": "m", "vel": "m/s", "beta_db": "dB (float16)", "rates": "Mbps"},
        "channel_H": "not stored; recreate from the seeds (see run_generate_dataset.py, walk_episode)",
    }
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# ------------------------------------------------------------------ checks
def verify(n_episodes, steps, kinds):
    D = np.load(MASTER)
    res = []

    def add(name, ok, detail):
        res.append((name, bool(ok), detail))
        print(f" [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("\nChecks on the master file")
    finite = all(np.isfinite(D[n].astype(np.float64)).all() for n in D.files)
    add("no NaN or infinite values", finite, f"{len(D.files)} arrays")
    add("complete: every episode has all its snapshots", len(D["episode"]) == n_episodes * steps and
        all((D["episode"] == e).sum() == steps for e in range(n_episodes)), f"{n_episodes} x {steps} = {n_episodes * steps} rows")
    add("label length is 121 x 7 + 100 = 947", D[f"label_{kinds[0]}"].shape[1] == LABEL_LEN, f"{D[f'label_{kinds[0]}'].shape}")
    add("traffic demand is equal (all 1.0) for every user", np.allclose(D["traffic"], TRAFFIC_DEMAND), f"min {D['traffic'].min()}, max {D['traffic'].max()}")
    add("users stay inside the 1 km x 1 km map", D["pos"].min() >= -1e-3 and D["pos"].max() <= cfg.AREA_M + 1e-3,
        f"x,y in [{D['pos'].min():.1f}, {D['pos'].max():.1f}] m")
    sp = np.linalg.norm(D["vel"], axis=2) * 3.6
    add("speeds stay inside 3-60 km/h", sp.min() >= 3 - 0.01 and sp.max() <= 60 + 0.01, f"[{sp.min():.2f}, {sp.max():.2f}] km/h")
    step_len = np.linalg.norm(np.diff(D["pos"].reshape(n_episodes, steps, K, 2), axis=1), axis=3)
    add("a user never moves farther than 60 km/h allows in one second", step_len.max() <= 60 / 3.6 * cfg.STEP_S + 0.05, f"largest step {step_len.max():.2f} m")
    for kd in kinds:
        lbl = D[f"label_{kd}"].astype(np.float64)
        p_ap = lbl[:, :M * L].reshape(-1, M, L).sum(axis=2)
        add(f"[{kd}] no AP uses more than its power budget", p_ap.max() <= 1 + 1e-5 and lbl.min() >= -1e-7, f"largest AP share {p_ap.max():.6f}")
        s = lbl[:, M * L:]
        add(f"[{kd}] scheduling shares sum to 1", np.abs(s.sum(axis=1) - 1).max() < 1e-4, f"worst {np.abs(s.sum(axis=1) - 1).max():.2e}")

    # reproducibility and label quality on a sample: recreate the channel from the seeds only
    pick_e = sorted(set([0, n_episodes // 2, n_episodes - 1]))
    pick_t = sorted(set([0, steps // 2, steps - 1]))
    ap_pos = layout.generate_ap_positions()
    channel = Eq2Channel(ap_pos)
    worst_pos = worst_beta = worst_rate = 0.0
    better = True
    n_chk = 0
    for e in pick_e:
        for t, pos, vel, snap in walk_episode(e, steps, channel):
            if t not in pick_t:
                continue
            row = e * steps + t
            worst_pos = max(worst_pos, np.abs(pos.astype(np.float32) - D["pos"][row]).max())
            worst_beta = max(worst_beta, np.abs(10 * np.log10(snap["beta"]) - D["beta_db"][row].astype(np.float64)).max())
            prob = lab.LabelProblem(snap["H"], snap["beta"], L)
            for kd in kinds:
                p = D[f"label_{kd}"][row][:M * L].astype(np.float64).reshape(M, L)
                r = prob.rates(p)
                worst_rate = max(worst_rate, (np.abs(r - D[f"rates_{kd}"][row]) / np.maximum(r, 1e-9)).max())
                f_label = lab.objective_and_weights(r, kd)[0]
                f_base = lab.objective_and_weights(prob.rates(lab.equal_power(prob, 3)), kd)[0]
                better &= f_label >= f_base - 1e-6 * K
            n_chk += 1
    add(f"reproducible: positions recreated from the seeds ({n_chk} sample snapshots)", worst_pos == 0, f"largest difference {worst_pos:.2e} m")
    add("reproducible: stored beta equals the recreated beta (float16 storage)", worst_beta < 0.1, f"largest difference {worst_beta:.3f} dB")
    add("reproducible: stored rates equal the rates of the label on the recreated channel", worst_rate < 1e-3, f"largest relative difference {worst_rate:.1e}")
    add("every sampled label scores at least as high as equal power (L=3)", better, "optimized >= baseline")
    with open(paths.table(f"dataset_{args.name}_checks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "result", "detail"])
        w.writerows([[n, "PASS" if ok else "FAIL", dtl] for n, ok, dtl in res])
    return all(ok for _, ok, _ in res), D


# ------------------------------------------------------------------ summary and picture
def summarize(D, kinds, meta, seconds):
    kd = kinds[0]
    r = D[f"rates_{kd}"]
    sp = np.linalg.norm(D["vel"], axis=2) * 3.6
    lbl = D[f"label_{kd}"][:, :M * L].reshape(-1, M, L)
    rows = [("episodes", meta["episodes"]), ("snapshots", meta["snapshots"]), ("users per snapshot", K), ("APs", M), ("label kinds", "+".join(kinds)),
            (f"mean speed of the label [{kd}] (Mbps)", f"{r.mean():.1f}"), ("slowest 5% (Mbps)", f"{np.percentile(r, 5):.1f}"),
            ("fastest 5% (Mbps)", f"{np.percentile(r, 95):.1f}"), ("users below 1 Mbps (%)", f"{(r < 1).mean() * 100:.3f}"),
            ("mean walking speed (km/h)", f"{sp.mean():.1f}"), ("beams switched on per AP (share > 1%)", f"{(lbl > 0.01).sum(axis=2).mean():.2f}"),
            ("master file size (MB)", f"{os.path.getsize(MASTER) / 1e6:.0f}"), ("generation time of this run (min)", f"{seconds / 60:.1f}")]
    with open(paths.table(f"dataset_{args.name}_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value"])
        w.writerows(rows)
    print("\nSummary")
    for a, b in rows:
        print(f" {a:45s} {b}")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].hist(r.ravel(), bins=60, color="#3b6fb6")
    axs[0, 0].set_title(f"Speed of every user under the label '{kd}' (Mbps)", fontsize=10)
    axs[0, 1].hist(sp.ravel(), bins=60, color="#c9772b")
    axs[0, 1].set_title("Walking speed of every user (km/h)", fontsize=10)
    per_t = r.reshape(meta["episodes"], meta["snapshots_per_episode"], K).mean(axis=(0, 2))
    axs[1, 0].plot(per_t, marker="o", ms=3)
    axs[1, 0].set_ylim(0, per_t.max() * 1.3)
    axs[1, 0].set_title("Mean speed at each second of the episode (flat = no drift)", fontsize=10)
    axs[1, 0].set_xlabel("second in the episode")
    axs[1, 1].hist((lbl > 0.01).sum(axis=2).ravel(), bins=np.arange(-0.5, L + 1.5), color="#4a9a6b", rwidth=0.85)
    axs[1, 1].set_title("Beams switched on per AP (of 7 slots)", fontsize=10)
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle(f"Master dataset '{args.name}': {meta['episodes']} episodes x {meta['snapshots_per_episode']} snapshots", fontsize=11)
    plt.tight_layout()
    plt.savefig(paths.image(f"dataset_{args.name}_overview.png"), dpi=110)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [e for e in range(args.episodes) if not is_finished(e, args.steps, KINDS)]
    print(f"Dataset '{args.name}': {args.episodes} episodes x {args.steps} snapshots, labels: {'+'.join(KINDS)}, workers: {args.workers}")
    print(f"{args.episodes - len(todo)} episodes are already finished, {len(todo)} to do.", flush=True)
    t0 = time.time()
    tasks = [(e, args.steps, args.iters, KINDS, OUT_DIR) for e in todo]
    done = 0
    if args.workers > 1 and tasks:
        with Pool(args.workers) as pool:
            for e, sec in pool.imap_unordered(make_episode, tasks):
                done += 1
                el = time.time() - t0
                print(f"episode {e:4d} done ({sec:.0f} s)   {done}/{len(tasks)}   elapsed {el / 60:.1f} min, about {el / done * (len(tasks) - done) / 60:.1f} min left", flush=True)
    else:
        for task in tasks:
            e, sec = make_episode(task)
            done += 1
            el = time.time() - t0
            print(f"episode {e:4d} done ({sec:.0f} s)   {done}/{len(tasks)}   elapsed {el / 60:.1f} min, about {el / done * (len(tasks) - done) / 60:.1f} min left", flush=True)
    meta = merge(args.episodes, args.steps, KINDS)
    ok, D = (True, np.load(MASTER)) if args.no_verify else verify(args.episodes, args.steps, KINDS)
    summarize(D, KINDS, meta, time.time() - t0)
    print(f"\nSaved {os.path.relpath(MASTER, paths.HERE)}, its meta file, tables/dataset_{args.name}_summary.csv, "
          f"tables/dataset_{args.name}_checks.csv, img/dataset_{args.name}_overview.png")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - do not use the file before they are understood")
    sys.exit(0 if ok else 1)