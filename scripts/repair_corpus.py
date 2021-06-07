import os
import random
import math
import sys

import project
from src.models.char_lm.unidirectional_model import UnidirectionalModel
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.helper.files import read_lines, write_lines


if __name__ == "__main__":
    #in_dir = "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/raw/"
    in_dir = "/nfs/datasets/tokenization-repair/acl-anthology/raw/"
    process_id = int(sys.argv[1])
    processes = 6
    random.seed(42)

    files = os.listdir(in_dir)
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

    out_dir = in_dir[:-1] + ".repaired/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file in files:
        print("** " + file + " **")
        sequences = read_lines(in_dir + file)
        repaired_sequences = []
        for sequence in sequences:
            sequence = sequence.strip()
            predicted = corrector.correct(sequence)
            if sequence != predicted:
                print(sequence)
                print(predicted)
            repaired_sequences.append(predicted)
        write_lines(out_dir + file, repaired_sequences)
