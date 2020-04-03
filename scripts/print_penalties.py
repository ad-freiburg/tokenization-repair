import project
from src.corrector.beam_search.penalty_holder import PenaltyHolder


if __name__ == "__main__":
    holder = PenaltyHolder()
    for key in holder.penalties:
        print(key, holder.penalties[key])
