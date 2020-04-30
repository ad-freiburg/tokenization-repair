import sys
import re
from wordsegment import load, segment
from unidecode import unidecode

import project
from src.interactive.sequence_generator import interactive_sequence_generator
from src.benchmark.benchmark import Benchmark, BenchmarkFiles, Subset
from src.postprocessing.rule_based import RuleBasedPostprocessor
from src.helper.time import time_diff, timestamp
from src.evaluation.predictions_file_writer import PredictionsFileWriter


def reinsert_punctuation(segmented: str, original: str) -> str:
    seg_i = orig_i = 0
    predicted = ""
    while seg_i < len(segmented) or orig_i < len(original):
        seg = segmented[seg_i] if seg_i < len(segmented) else None
        orig = original[orig_i] if orig_i < len(original) else None
        if seg == orig:
            predicted += orig
            seg_i, orig_i = seg_i + 1, orig_i + 1
        elif seg is not None and orig is not None and unidecode(seg) == unidecode(orig).lower():
            predicted += orig
            seg_i, orig_i = seg_i + 1, orig_i + 1
        elif orig is not None and re.match("\W", orig) and seg is not None and seg == ' ':
            if len(predicted) > 0 and predicted[-1] != ' ':
                predicted += ' '
            predicted += orig
            if orig_i + 1 < len(original):
                predicted += ' '
            orig_i += 1
        elif seg == ' ':
            if len(predicted) > 0 and predicted[-1] != ' ':
                predicted += seg
            seg_i += 1
        elif orig is not None:
            predicted += orig
            orig_i += 1
        else:
            predicted += seg
            seg_i += 1
    return predicted


class WordSegment:
    def __init__(self):
        load()
        self.postprocessor = RuleBasedPostprocessor()

    def correct(self, sequence: str) -> str:
        sequence = ''.join(sequence.split())
        segmented = ' '.join(segment(sequence))
        #print(segmented)
        predicted = reinsert_punctuation(segmented, sequence)
        #print(predicted)
        predicted = self.postprocessor.correct(predicted)
        return predicted


if __name__ == "__main__":
    if len(sys.argv) > 1:
        benchmark_name = sys.argv[1]
        benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        writer = PredictionsFileWriter(benchmark.get_results_directory() + "wordsegment.txt")
    else:
        sequences = interactive_sequence_generator()
        writer = None

    segmenter = WordSegment()

    for s_i, sequence in enumerate(sequences):
        start_time = timestamp()
        predicted = segmenter.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if writer is not None:
            writer.add(predicted, runtime)

    if writer is not None:
        writer.save()
