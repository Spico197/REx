import sys

from .cmd import startproject
from .cmd import clean


if __name__ == "__main__":
    arg = sys.argv[1]
    arg2action = {
        "startproject": startproject,
        "clean": clean
    }
    arg2action[arg]()
