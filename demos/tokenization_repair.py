from typing import Optional

import sys
import argparse

import project
from src.interactive.sequence_generator import interactive_sequence_generator
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.corrector.beam_search.batched_beam_search_corrector import BatchedBeamSearchCorrector
from src.corrector.beam_search.penalty_holder import PenaltyHolder


APPROACHES = ("UNI", "BID", "BID+", "COMBO+", "ONE")
PENALTIES = ("Wiki", "Wiki_typos", "ACL", "arXiv")
APPROACHES2MODELS = {
    "UNI": ("fwd1024", None),
    "BID": ("fwd1024", "labeling_ce"),
    "BID+": ("fwd1024_noise0.2", "labeling_noisy_ce"),
    "COMBO+": ("combined_mixed_forward_robust", "combined_mixed_labeling_robust"),
    "ONE": ("combined_mixed_forward_robust", "combined_mixed_labeling_robust")
}
PENALTIES2BENCHMARKS = {
    "Wiki": "0_0.1",
    "Wiki_typos": "0.1_0.1",
    "ACL": "nastase-big",
    "arXiv": "arxiv-910k"
}


def get_corrector(approach: str, penalties: Optional[str], insertion_penalty: Optional[float], deletion_penalty: Optional[float]):
    fwd_model_name, bid_model_name = APPROACHES2MODELS[approach]
    fwd_model = UnidirectionalLMEstimator()
    fwd_model.load(fwd_model_name)
    if bid_model_name is None:
        bid_model = None
    else:
        bid_model = BidirectionalLabelingEstimator()
        bid_model.load(bid_model_name)
    if approach == "ONE":
        if benchmark is not None and benchmark.endswith("no_spaces"):
            p_ins = p_del = 0
        else:
            holder = PenaltyHolder(seq_acc=True)
            model_key = fwd_model_name + "_" + bid_model_name
            p_ins, p_del = holder.get(model_key, "nastase-big")
    elif penalties is None:
        p_ins = p_del = 0
    else:
        holder = PenaltyHolder(seq_acc=True)
        model_key = fwd_model_name
        if bid_model_name is not None:
            model_key += "_" + bid_model_name
        benchmark_key = PENALTIES2BENCHMARKS[penalties]
        p_ins, p_del = holder.get(model_key, benchmark_key)
    if insertion_penalty is not None:
        p_ins = -insertion_penalty
    if deletion_penalty is not None:
        p_del = -deletion_penalty
    corrector = BatchedBeamSearchCorrector(fwd_model, insertion_penalty=p_ins, deletion_penalty=p_del, n_beams=5,
                                           verbose=benchmark is None, labeling_model=bid_model,
                                           add_epsilon=bid_model is not None)
    return corrector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Run the tokenization repair interactively or on one of the benchmarks. "
                                     "Models, data and benchmarks must be located at /external. "
                                     "You can specify a benchmark from /external/benchmarks "
                                     "and add your own benchmarks to that directory.")
    parser.add_argument("-a", dest="approach", type=str, choices=APPROACHES,
                        required=False, default="ONE",
                        help="Select the tokenization repair method (default: ONE).")
    parser.add_argument("-p", dest="penalties", type=str, choices=PENALTIES, required=False,
                        help="Choose penalties optimized for one of the benchmarks (default: no penalties).")
    parser.add_argument("-p_ins", dest="p_ins", type=float, required=False, default=None,
                        help="Set the insertion penalty explicitely (default: None).")
    parser.add_argument("-p_del", dest="p_del", type=float, required=False, default=None,
                        help="Set the deletion penalty explicitely (default: None).")
    parser.add_argument("-b", dest="benchmark", type=str, required=False,
                        help="Select a benchmark to run the approach on (default: run in interactive console).")
    parser.add_argument("--test", action="store_true",
                        help="Run the approach on the test set of the selected benchmark (default: development set).")
    parser.add_argument("-f", dest="out_file", type=str, required=False,
                        help="Specify a file name to save your results (only when a benchmark is selected).")
    args = parser.parse_args()

    approach = args.approach
    penalties = args.penalties
    p_ins = args.p_ins
    p_del = args.p_del
    benchmark = args.benchmark
    test = args.test
    out_file = args.out_file
    print("approach:", approach)
    print("penalties:", penalties)
    print("p_ins:", p_ins)
    print("p_del:", p_del)
    print("benchmark:", benchmark)
    print("test:", test)
    print("out file:", out_file)

    corrector = get_corrector(approach, penalties, p_ins, p_del)
    print("P_ins = %.1f" % -corrector.insertion_penalty)
    print("P_del = %.1f" % -corrector.deletion_penalty)

    if benchmark is None:
        sequences = interactive_sequence_generator()
        file_writer = None
    else:
        subset = Subset.TEST if test else Subset.DEVELOPMENT
        benchmark = Benchmark(benchmark, subset)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        if out_file is not None:
            file_writer = PredictionsFileWriter(benchmark.get_results_directory() + out_file)

    for sequence in sequences:
        if sequence.startswith("#"):
            if out_file is not None:
                file_writer.add(sequence, 0)
            continue
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if out_file is not None:
            file_writer.add(predicted, runtime)
            file_writer.save()
