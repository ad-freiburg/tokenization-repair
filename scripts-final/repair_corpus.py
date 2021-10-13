import os
import random
import math
import argparse
import numpy as np

import project
from src.models.char_lm.unidirectional_model import UnidirectionalModel
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.helper.files import read_lines, write_lines


def main(args):
    in_dir = args.input_directory
    process_id = 0
    processes = 1
    random.seed(42)

    if not in_dir.endswith("/"):
        in_dir += "/"

    files = os.listdir(in_dir)

    if processes > 1:
        random.shuffle(files)
        files_per_process = math.ceil(len(files) / processes)
        files = sorted(files[(process_id * files_per_process):((process_id + 1) * files_per_process)])

    uni_model = UnidirectionalModel(args.fwd_model)

    if args.bid_model == "None":
        bid_model = None
    else:
        bid_model = BidirectionalLabelingEstimator()
        bid_model.load(args.bid_model)

    corrector = BatchedBeamSearchCorrector(
        uni_model.model,
        insertion_penalty=-args.p_ins,
        deletion_penalty=-args.p_del,
        n_beams=5,
        verbose=False,
        labeling_model=bid_model,
        add_epsilon=True
    )

    out_dir = in_dir[:-1] + ".repaired/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file in files:
        print("** " + file + " **")
        sequences = read_lines(in_dir + file)
        sequences = [s.strip() for s in sequences]
        repaired_sequences = []
        i = 0
        while i < len(sequences):
            batch_sequences = [sequences[i]]
            if args.concat_hyphenated:
                while batch_sequences[-1][-1] == "-" and i + 1 < len(sequences):
                    i += 1
                    batch_sequences.append(sequences[i])
            i += 1
            batch = "".join(batch_sequences)
            predicted = corrector.correct(batch)
            if predicted != batch:
                print(predicted)
            if len(batch_sequences) == 1:
                repaired_sequences.append(predicted)
            else:
                split_positions = set(np.cumsum([len(seq.replace(" ", "")) for seq in batch_sequences]))
                start = 0
                nospace_chars = 0
                for pos in range(len(predicted)):
                    if predicted[pos] != " ":
                        nospace_chars += 1
                        if nospace_chars in split_positions:
                            seq = predicted[start:(pos + 1)].strip()
                            repaired_sequences.append(seq)
                            start = pos + 1
        write_lines(out_dir + file, repaired_sequences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair the tokenization of all files in a given directory.")
    parser.add_argument("input_directory", type=str,
                        help="A directory containing text files which will be repaired.")
    parser.add_argument("--no_concat_hyphenated", dest="concat_hyphenated", action="store_false",
                        help="Do not concatenate lines separated with a hyphen + newline, "
                             "but repair each line independently.")
    parser.add_argument("--fwd_model", type=str, default="conll.fwd1024.ocr+spelling",
                        help="Name of the unidirectional model.")
    parser.add_argument("--bid_model", type=str, default="conll.fwd1024.ocr+spelling",
                        help="Name of the bidirectional model ('None' to use no bidirectional model).")
    parser.add_argument("--p_ins", type=float, default=7.6,
                        help="Insertion penalty (>= 0).")
    parser.add_argument("--p_del", type=float, default=9.2,
                        help="Deletion penalty (>= 0).")
    parser.set_defaults(concat_hyphenated=True)
    args = parser.parse_args()
    main(args)
