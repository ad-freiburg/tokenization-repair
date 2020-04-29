import sys
import math

from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

import project
from src.helper.pickle import dump_object, load_object
from src.settings import paths
from src.sequence.sentence_splitter import NLTKSentenceSplitter


TEXT_FILE = paths.WIKI_DIR + "single-file/training_shuffled.txt"
PICKLE_FILE = paths.WIKI_PUNKT_TOKENIZER


def get_text(n: int) -> str:
    text = ""
    for i, line in enumerate(open(TEXT_FILE)):
        if i == n:
            break
        text += line
    return text


def score(trainer: PunktTrainer, typ: str) -> float:
    # Count how many periods & nonperiods are in the
    # candidate.
    num_periods = typ.count('.') + 1
    num_nonperiods = len(typ) - num_periods + 1

    # Let <a> be the candidate without the period, and <b>
    # be the period.  Find a log likelihood ratio that
    # indicates whether <ab> occurs as a single unit (high
    # value of ll), or as two independent units <a> and
    # <b> (low value of ll).
    count_with_period = trainer._type_fdist[typ + '.']
    count_without_period = trainer._type_fdist[typ]
    ll = trainer._dunning_log_likelihood(
        count_with_period + count_without_period,
        trainer._num_period_toks,
        count_with_period,
        trainer._type_fdist.N(),
    )

    # Apply three scaling factors to 'tweak' the basic log
    # likelihood ratio:
    #   F_length: long word -> less likely to be an abbrev
    #   F_periods: more periods -> more likely to be an abbrev
    #   F_penalty: penalize occurrences w/o a period
    f_length = math.exp(-num_nonperiods)
    f_periods = num_periods
    f_penalty = int(trainer.IGNORE_ABBREV_PENALTY) or math.pow(
        num_nonperiods, -count_without_period
    )
    score = ll * f_length * f_periods * f_penalty
    return score


if __name__ == "__main__":
    MODE = sys.argv[1]

    if MODE == "train":
        n = int(sys.argv[2])

        print("reading...")
        text = get_text(n)

        print("training...")
        trainer = PunktTrainer()
        trainer.INCLUDE_ALL_COLLOCS = True
        trainer.ABBREV = 0.3
        trainer.train(text, verbose=True)
        del text

        print("building tokenizer...")
        tokenizer = PunktSentenceTokenizer(trainer.get_params())
        abbrevs = tokenizer._params.abbrev_types
        print(sorted(abbrevs))
        print("%i abbreviations" % len(abbrevs))

        target_abbrevs = ["i.e", "e.g", "prof", "dr", "m.sc", "no", "nos", "mr", "mrs", "ms", "seq", "o.r.s"]
        for target in target_abbrevs:
            print(target, target in abbrevs, score(trainer, target))

        print("saving...")
        dump_object(tokenizer, PICKLE_FILE)

    elif MODE == "starters":
        tokenizer = load_object(PICKLE_FILE)
        for starter in sorted(tokenizer._params.sent_starters):
            print(starter)

    elif MODE == "starters-nltk":
        tokenizer = NLTKSentenceSplitter()
        for starter in sorted(tokenizer.tokenizer._params.sent_starters):
            print(starter)