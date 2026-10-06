# Turns the STARTING positions of the users into a whole walk:
# one position per user and per time step.

import numpy as np

import config


def unfold(x, area):
    """Bounce off the walls of the map.

    A tempting but WRONG way: take one step, and if that puts you outside the
    map, mirror that one point back inside. It only gives the right answer for
    steps that do not cross a wall.
    The right way, used here: imagine the user walks in one unbroken straight
    line (start + velocity x time) and then FOLD that whole line into the map
    like a zig-zag. This is the standard exact way to describe a ball bouncing
    around a rectangular box, and it is correct at every moment."""
    m = np.mod(x, 2 * area)
    return np.where(m <= area, m, 2 * area - m)


def generate_trajectories(start_pos, seed=config.SEED):
    """Every user gets a random walking speed and a random direction, and walks
    in a straight line for config.N_STEPS steps, bouncing off the walls.

    Returns
    -------
    pos   : (N_STEPS, N_UE, 2)  [x, y] of every user at every time step (metres)
    speed : (N_UE,)             walking speed of every user (m/s)
    """
    n_ue = len(start_pos)
    rng = np.random.default_rng(seed)
    speed = rng.uniform(config.SPEED_MIN_MS, config.SPEED_MAX_MS, n_ue)
    angle = rng.uniform(0, 2 * np.pi, n_ue)
    velocity = np.stack([speed * np.cos(angle), speed * np.sin(angle)], axis=1)

    t = np.arange(config.N_STEPS) * config.DT_S
    straight_line = start_pos[None, :, :] + velocity[None, :, :] * t[:, None, None]
    return unfold(straight_line, config.AREA_M), speed


if __name__ == "__main__":
    from ue_location_generation import generate_ue_positions
    pos, speed = generate_trajectories(generate_ue_positions())
    print("positions shape:", pos.shape, "(time steps, users, x/y)")
    print("speeds (m/s):", speed.round(2))