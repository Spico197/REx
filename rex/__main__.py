import sys

from .cmd import new_project
from .cmd import clean_project


if __name__ == "__main__":
    arg = sys.argv[1]
    arg2action = {
        "new": new_project,
        "clean": clean_project
    }
    arg2action[arg]()
