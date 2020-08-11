import sys

import project
from src.corrector.load.beam_search import load_beam_search_corrector, load_two_pass_corrector
from src.corrector.load.labeling import load_labeling_corrector
from src.interactive.sequence_generator import interactive_sequence_generator
from src.benchmark.benchmark import Benchmark, SUBSETS, BenchmarkFiles
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp


def print_help():
    print("Usage:")
    print("    python3 demos/tokenization_repair.py <approach> <robust> <typos> <p> "
          "[<benchmark> <subset> <output file>]")
    print()
    print("Arguments:")
    print("    <approach>: Choose from {bidir, BS-fw, BS-bw, 2-pass, BS-bidir}.")
    print("    <robust>:   Set to 'robust' for robust models, anything else for non-robust models.")
    print("    <typos>:    Set to 'typos' for the penalties optimized with typos, anything else otherwise.")
    print("    <p>:        Pick the penalties for a specific setting. Choose from {0.1, 1, inf} to specify the " 
          "tokenization error rate (inf for the no-spaces scenario).")
    print()
    print("Optional arguments:")
    print("    <benchmark>: Select a benchmark from DATA/benchmarks/.")
    print("        0_0.1:   no typos, 0.1 tokenization error rate")
    print("        0_1:     no typos, 1.0 tokenization error rate")
    print("        0_inf:   no typos, no spaces")
    print("        0.1_0.1: with typos, 0.1 tokenization error rate")
    print("        0.1_1:   with typos, 1.0 tokenization error rate")
    print("        0.1_inf: with typos, no spaces")
    print("        doval:   English benchmark from [Doval and Gómez-Rodríguez, 2019]")
    print("        -- your own benchmark here --")
    print("    <subset>: Choose a benchmark subset from {development, test}.")
    print("    <output file>: File name for the predicted sequences.")
    print("        You will find the file in DATA/results/<benchmark>/<subset>/.")
    print()
    print("Example:")
    print("    python3 demos/tokenization_repair.py BS-bidir robust typos 0.1 0.1_0.1 test BS-bidir.txt")


APPROACHES = {"bidir", "BS-fw", "BS-bw", "2-pass", "BS-bidir"}
P_LEVELS = {0.1, 1, float("inf")}

N_ARGS_EXPECTED = {4, 7}


def get_arguments():
    n_args = len(sys.argv) - 1
    if n_args not in N_ARGS_EXPECTED:
        if n_args > 0:
            args_expected_str = " or ".join(str(n) for n in sorted(N_ARGS_EXPECTED))
            print("ERROR: please specify %s arguments. Found %i." % (args_expected_str, n_args))
        exit(1)
    approach = sys.argv[1]
    if approach not in APPROACHES:
        print("ERROR: please specify an approach from %s." % str(APPROACHES))
        exit(1)
    robust = sys.argv[2] == 'robust'
    typos = sys.argv[3] == 'typos'
    try:
        p = float(sys.argv[4])
        if p not in P_LEVELS:
            raise Exception()
    except:
        print("ERROR: p must be in %s." % str(P_LEVELS))
        exit(1)
    benchmark = subset = file_name = None
    if n_args == 7:
        benchmark = sys.argv[5]
        subset = sys.argv[6]
        if subset not in SUBSETS:
            print("ERROR: subset must be in %s." % str(set(key for key in SUBSETS)))
            exit(1)
        subset = SUBSETS[subset]
        file_name = sys.argv[7]
    return approach, robust, typos, p, benchmark, subset, file_name


def get_corrector(approach: str, robust: bool, typos: bool, p: float):
    bidir = approach == "bidir"
    two_pass = approach == "2-pass"
    if bidir:
        corrector = load_labeling_corrector(robust, typos, p)
    elif two_pass:
        corrector = load_two_pass_corrector(robust, typos, p)
    else:
        backward = approach == "BS-bw"
        bidir = approach == "BS-bidir"
        corrector = load_beam_search_corrector(backward, robust, typos, p, bidir)
    return corrector


if __name__ == "__main__":
    if len(sys.argv) == 1 or "-h" in sys.argv or "-help" in sys.argv or "help" in sys.argv:
        print_help()
        exit(0)

    approach, robust, typos, p, benchmark, subset, file_name = get_arguments()

    corrector = get_corrector(approach, robust, typos, p)

    if benchmark is None:
        sequences = interactive_sequence_generator()
        file_writer = None
    else:
        benchmark = Benchmark(benchmark, subset)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + file_name)

    for sequence in sequences:
        if sequence.startswith("#"):
            if file_writer is not None:
                file_writer.add(sequence, 0)
            continue
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        if file_writer is not None:
            file_writer.add(predicted, runtime)
            file_writer.save()
