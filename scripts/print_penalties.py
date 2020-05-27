import sys

import project
from src.corrector.beam_search.penalty_holder import PenaltyHolder


if __name__ == "__main__":
    holder = PenaltyHolder(two_pass="2-pass" in sys.argv)
    for key in sorted(holder.penalties):
        print(key, holder.penalties[key])
