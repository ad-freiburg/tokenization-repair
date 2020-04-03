from project import src
from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("benchmark", "-benchmark", "str",
              help_message="Name of the benchmark."),
    Parameter("test", "-t", "boolean"),
    Parameter("file", "-f", "str",
              help_message="Name of the file containing predicted sequences.")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.evaluation.evaluator import Evaluator
from src.helper.data_structures import izip
from src.evaluation.print_methods import print_evaluator


if __name__ == "__main__":
    benchmark_name = parameters["benchmark"]
    benchmark_subset = Subset.TEST if parameters["test"] else Subset.DEVELOPMENT
    benchmark = Benchmark(benchmark_name,
                          subset=benchmark_subset)

    sequence_pairs = benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT)
    predicted_sequences = benchmark.get_predicted_sequences(parameters["file"])

    evaluator = Evaluator()

    for s_i, (correct, corrupt), predicted in izip(sequence_pairs, predicted_sequences):
        evaluator.evaluate(benchmark,
                           s_i,
                           correct,
                           corrupt,
                           predicted,
                           evaluate_ed=False)

        evaluator.print_sequence()

    print_evaluator(evaluator)
