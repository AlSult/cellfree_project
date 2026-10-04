# Decides WHERE the APs stand.

import numpy as np


def generate_ap_positions():
    """Returns a table with one row per AP: [x, y] in metres."""
    return np.array([[50.0, 50.0],      # AP 0
                     [150.0, 150.0]])   # AP 1


if __name__ == "__main__":
    print(generate_ap_positions())