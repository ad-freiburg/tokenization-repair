import sys

import project
from src.corrector.beam_search.penalty_holder import PenaltyHolder


if __name__ == "__main__":
    holder = PenaltyHolder(seq_acc=True)
    for key in sorted(holder.penalties):
        print(key, holder.penalties[key])
