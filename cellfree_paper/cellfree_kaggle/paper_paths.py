# Where everything lives. All the paths are relative to THIS folder (cellfree_paper), so the
# scripts give the same result wherever you start them from:
#     img/     the pictures
#     tables/  the tables (CSV files)
#     data/    the saved snapshots
# The folders are created automatically.

import os

HERE = os.path.dirname(os.path.abspath(__file__))     # cellfree_paper
PARENT = os.path.dirname(HERE)                         # cellfree_project (the folder that holds this one)
IMG_DIR = os.path.join(HERE, "img")
TABLE_DIR = os.path.join(HERE, "tables")
DATA_DIR = os.path.join(HERE, "data")


def _in(folder, name):
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def image(name):
    return _in(IMG_DIR, name)


def table(name):
    return _in(TABLE_DIR, name)


def data(name):
    return _in(DATA_DIR, name)


def find_middle_results():
    """step8a_results.csv, the results table of the middle network. It lives in cellfree_project,
    the folder above this one. Returns None if it cannot be found."""
    for folder in (PARENT, HERE, TABLE_DIR):
        path = os.path.join(folder, "step8a_results.csv")
        if os.path.exists(path):
            return path
    return None


if __name__ == "__main__":
    print("pictures :", IMG_DIR)
    print("tables   :", TABLE_DIR)
    print("snapshots:", DATA_DIR)
    print("middle-network results found:", find_middle_results())