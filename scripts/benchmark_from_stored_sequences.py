from project import src
from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("benchmark", "-b", "str",
              help_message="Name of the benchmark."),
    Parameter("test", "-t", "boolean"),
    Parameter("file", "-f", "str",
              help_message="Name of the file containing predicted sequences."),
    Parameter("save", "-s", "str")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.evaluation.evaluator import Evaluator
from src.helper.data_structures import izip
from src.evaluation.print_methods import print_evaluator
from src.evaluation.results_holder import ResultsHolder, Metric


if __name__ == "__main__":
    benchmark_name = parameters["benchmark"]
    benchmark_subset = Subset.TEST if parameters["test"] else Subset.DEVELOPMENT
    benchmark = Benchmark(benchmark_name,
                          subset=benchmark_subset)

    sequence_pairs = benchmark.get_sequence_pairs(BenchmarkFiles.CORRUPT)
    if parameters["file"] == "corrupt.txt":
        predicted_sequences = [corrupt for _, corrupt in sequence_pairs]
    else:
        predicted_sequences = benchmark.get_predicted_sequences(parameters["file"])

    evaluator = Evaluator()
    results_holder = ResultsHolder()

    for s_i, (correct, corrupt), predicted in izip(sequence_pairs, predicted_sequences):
        evaluator.evaluate(benchmark,
                           s_i,
                           correct,
                           corrupt,
                           predicted,
                           evaluate_ed=False)
        evaluator.print_sequence()

    print_evaluator(evaluator)

    if parameters["save"] != "0":
        f1 = evaluator.f1()
        acc = evaluator.sequence_accuracy()
        results_holder.set(benchmark_name, benchmark_subset, parameters["save"],
                           [(Metric.F1, f1), (Metric.SEQUENCE_ACCURACY, acc)])
        results_holder.save()
