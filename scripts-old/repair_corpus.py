import os
import random
import math
import sys

import numpy as np

import project
from src.models.char_lm.unidirectional_model import UnidirectionalModel
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.helper.files import read_lines, write_lines


if __name__ == "__main__":
    in_dir = sys.argv[1]  # "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/raw/"
    process_id = 0  # int(sys.argv[1])
    processes = 1
    random.seed(42)

    if not in_dir.endswith("/"):
        in_dir += "/"

    files = os.listdir(in_dir)

    if processes > 1:
        random.shuffle(files)
        files_per_process = math.ceil(len(files) / processes)
        files = sorted(files[(process_id * files_per_process):((process_id + 1) * files_per_process)])

    uni_model = UnidirectionalModel("conll.fwd1024.ocr+spelling")
    bid_model = BidirectionalLabelingEstimator()
    bid_model.load("conll.labeling.ocr+spelling")

    corrector = BatchedBeamSearchCorrector(
        uni_model.model,
        insertion_penalty=-7.6,
        deletion_penalty=-9.2,
        n_beams=5,
        verbose=False,
        labeling_model=bid_model,
        add_epsilon=True
    )

    out_dir = in_dir[:-1] + ".repaired_hyphens/"
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
            while batch_sequences[-1][-1] == "-" and i + 1 < len(sequences):
                i += 1
                batch_sequences.append(sequences[i])
            i += 1
            batch = "".join(batch_sequences)
            predicted = corrector.correct(batch)
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
