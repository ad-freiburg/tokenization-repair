from project import src
from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("benchmark", "-b", "str",
              help_message="Name of the benchmark."),
    Parameter("set", "-set", "str"),
    Parameter("sequences", "-n", "int"),
    Parameter("file", "-f", "str",
              help_message="Name of the file containing predicted sequences.")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.benchmark.benchmark import Benchmark, BenchmarkFiles, SUBSETS
from src.evaluation.evaluator import Evaluator
from src.helper.data_structures import izip
from src.evaluation.print_methods import print_evaluator


if __name__ == "__main__":
    benchmark_name = parameters["benchmark"]
    benchmark_subset = SUBSETS[parameters["set"]]
    benchmark = Benchmark(benchmark_name,
                          subset=benchmark_subset)

    sequence_pairs = benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT)
    if parameters["file"] == "corrupt.txt":
        predicted_sequences = [corrupt for _, corrupt in sequence_pairs]
    else:
        predicted_sequences = benchmark.get_predicted_sequences(parameters["file"])

    evaluator = Evaluator()

    for s_i, (correct, corrupt), predicted in izip(sequence_pairs, predicted_sequences):
        if s_i == parameters["sequences"]:
            break
        evaluator.evaluate(benchmark,
                           s_i,
                           correct,
                           corrupt,
                           predicted,
                           evaluate_ed=False)
        evaluator.print_sequence()

    print_evaluator(evaluator)
