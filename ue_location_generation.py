# Decides where the users (UEs) START.

import numpy as np


def generate_ue_positions():
    """Returns a table with one row per user: [x, y] in metres."""
    return np.array([[60.0, 60.0],      # User 0: close to AP 0
                     [100.0, 100.0],    # User 1: exactly in the middle
                     [140.0, 140.0],    # User 2: close to AP 1
                     [20.0, 180.0]])    # User 3: far from both


if __name__ == "__main__":
    print(generate_ue_positions())