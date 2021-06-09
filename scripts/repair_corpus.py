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
    in_dir = "/home/hertel/tokenization-repair-dumps/nastase/acl-201302_word-resegmented/raw/"
    #in_dir = "/nfs/datasets/tokenization-repair/acl-anthology/raw/"
    process_id = int(sys.argv[1])
    processes = 5
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

    out_dir = in_dir[:-1] + ".repaired_three/"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for file in files:
        print("** " + file + " **")
        sequences = read_lines(in_dir + file)[:5]
        sequences = [s.strip() for s in sequences]
        repaired_sequences = []
        for i in range(len(sequences)):
            start = i  # max(0, i - 1)
            end = i + 1
            paragraph = ""
            undeletable_space_positions = set()
            for paragraph_sequence in sequences[start:(end + 1)]:
                if len(paragraph) > 0 and not paragraph.endswith("-"):
                    undeletable_space_positions.add(len(paragraph.replace(" ", "")))
                    paragraph += " "
                paragraph += paragraph_sequence
            predicted = corrector.correct(paragraph, undeletable_space_positions=undeletable_space_positions)
            prefix_len_no_spaces = 0 if start == i else len(sequences[i - 1].replace(" ", ""))
            seq_len_no_spaces = len(sequences[i].replace(" ", ""))
            c_no_space_chars = 0
            for j, char in enumerate(predicted):
                if c_no_space_chars == prefix_len_no_spaces:
                    span_start = j
                if char != " ":
                    c_no_space_chars += 1
                if c_no_space_chars == prefix_len_no_spaces + seq_len_no_spaces:
                    span_end = j + 1
                    break
            print(undeletable_space_positions)
            predicted_sequence = predicted[span_start:span_end]
            if sequences[i] != predicted_sequence:
                print(sequences[i])
                print(paragraph)
                print(predicted)
                print(predicted_sequence)
            repaired_sequences.append(predicted_sequence)
        write_lines(out_dir + file, repaired_sequences)
