# Random Waypoint walking (Table I of the paper: Random Waypoint, 3 to 60 km/h).
#
# Every user repeats:  pick a random destination in the area, walk there in a straight line at a
# constant speed picked at random between 3 and 60 km/h, (pause), pick the next destination.
#
# Two ways to START the walk:
#   "stationary"  the exact steady state of the process (Navidi and Camp). Users are already spread the
#                 way they would be after hours of walking, and the speeds are already the right mix of
#                 slow and fast. THIS IS THE DEFAULT.
#   "naive"       uniform start positions and a fresh random speed. It looks natural, but it is NOT the
#                 steady state: slow legs last longer, so the average speed of the crowd falls for the first
#                 minutes (the well-known "speed decay" of Random Waypoint).

import numpy as np

import paper_config as cfg


def _stationary_first_leg(rng, side, v_min, v_max):
    """First leg drawn from the time-stationary distribution of Random Waypoint (pause = 0):
       - the two end points of a leg are drawn with probability proportional to the leg length
         (long legs last longer, so a user is more likely to be on one);
       - the speed is drawn with density proportional to 1 / v (slow legs last longer);
       - the user is at a uniformly random point of the leg."""
    longest = side * np.sqrt(2.0)
    while True:
        a, b = rng.uniform(0, side, 2), rng.uniform(0, side, 2)
        if rng.uniform() < np.linalg.norm(a - b) / longest:
            break
    v = v_min * (v_max / v_min) ** rng.uniform()
    start = a + rng.uniform() * (b - a)
    return start, b, v


def _one_user(rng, n_steps, dt, side, v_min, v_max, pause_s, start, burn_in_s):
    """Positions, velocities and speeds of one user at the times 0, dt, 2 dt, ..."""
    offset = burn_in_s if (pause_s > 0 and start == "stationary") else 0.0
    if start == "stationary" and pause_s == 0:
        p, dest, v = _stationary_first_leg(rng, side, v_min, v_max)
    else:
        p, dest, v = rng.uniform(0, side, 2), rng.uniform(0, side, 2), rng.uniform(v_min, v_max)
    t_end = offset + (n_steps - 1) * dt
    t0, p0, p1, speed, duration = [], [], [], [], []
    t = 0.0
    while t <= t_end:
        d = np.linalg.norm(dest - p)
        t0.append(t); p0.append(p.copy()); p1.append(dest.copy()); speed.append(v); duration.append(d / v)
        t += d / v
        p = dest.copy()
        if pause_s > 0:                                   # stand still for a while
            t0.append(t); p0.append(p.copy()); p1.append(p.copy()); speed.append(0.0); duration.append(pause_s)
            t += pause_s
        dest, v = rng.uniform(0, side, 2), rng.uniform(v_min, v_max)
    t0, p0, p1, speed, duration = map(np.array, (t0, p0, p1, speed, duration))
    times = offset + np.arange(n_steps) * dt
    leg = np.searchsorted(t0, times, side="right") - 1
    frac = np.clip((times - t0[leg]) / np.maximum(duration[leg], 1e-12), 0.0, 1.0)
    pos = p0[leg] + (p1[leg] - p0[leg]) * frac[:, None]
    direction = (p1[leg] - p0[leg]) / np.maximum(np.linalg.norm(p1[leg] - p0[leg], axis=1, keepdims=True), 1e-12)
    vel = direction * speed[leg][:, None]
    return pos, vel, speed[leg], leg


def random_waypoint(n_users, n_steps, rng, dt=cfg.STEP_S, side=cfg.AREA_M, v_min=cfg.RWP_V_MIN_MS,
                    v_max=cfg.RWP_V_MAX_MS, pause_s=cfg.RWP_PAUSE_S, start="stationary", burn_in_s=cfg.RWP_BURN_IN_S):
    """Walk n_users for n_steps snapshots dt seconds apart.
    Returns   pos (n_steps, n_users, 2) in metres, vel (n_steps, n_users, 2) in m/s,
              speed (n_steps, n_users) in m/s, leg (n_steps, n_users) number of the leg each user is on."""
    pos = np.zeros((n_steps, n_users, 2))
    vel = np.zeros((n_steps, n_users, 2))
    speed = np.zeros((n_steps, n_users))
    leg = np.zeros((n_steps, n_users), dtype=int)
    for k in range(n_users):
        pos[:, k], vel[:, k], speed[:, k], leg[:, k] = _one_user(rng, n_steps, dt, side, v_min, v_max, pause_s, start, burn_in_s)
    return pos, vel, speed, leg


def stationary_positions(n_users, rng, side=cfg.AREA_M):
    """Positions of n_users at one instant of the steady state (what a 'static drop' looks like for Random Waypoint)."""
    return np.array([_stationary_first_leg(rng, side, cfg.RWP_V_MIN_MS, cfg.RWP_V_MAX_MS)[0] for _ in range(n_users)])


# ---------------- theory, used by the audit ----------------
def theory_time_average_speed(v_min=cfg.RWP_V_MIN_MS, v_max=cfg.RWP_V_MAX_MS):
    """Time-average speed of a Random Waypoint user with speed uniform in [v_min, v_max]: 1 / E[1/v]."""
    return (v_max - v_min) / np.log(v_max / v_min)


def theory_mean_leg_length(side=cfg.AREA_M):
    """Mean distance between two uniform points of a square (the leg length): 0.5214 x side."""
    return side * (2 + np.sqrt(2) + 5 * np.log(1 + np.sqrt(2))) / 15


def theory_central_share():
    """Approximate share of users in the middle quarter (half the side in x and in y) in the steady state,
    from the closed-form approximation of Bettstetter et al. (2003): marginal density 6/a^3 (a^2/4 - x^2)."""
    one_axis = 6 * (0.25 * 0.5 - 2 * 0.25 ** 3 / 3)                    # integral of the marginal over |x| <= a/4 (a = 1)
    return one_axis ** 2


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pos, vel, speed, leg = random_waypoint(5, 10, rng)
    print("positions", pos.shape, " speeds (m/s) of the first step:", speed[0].round(2))
    print("theory: time-average speed", round(theory_time_average_speed(), 3), "m/s,  mean leg", round(theory_mean_leg_length()), "m,  central share", round(theory_central_share(), 3))