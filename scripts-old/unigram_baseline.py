import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("words", "-w", "int"),
          Parameter("benchmark", "-b", "str"),
          Parameter("test", "-t", "boolean")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.interactive.sequence_generator import interactive_sequence_generator
from src.benchmark.benchmark import Benchmark, BenchmarkFiles, Subset
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp
from src.ngram.unigram_corrector import UnigramCorrector


if __name__ == "__main__":
    n = parameters["words"]
    n = None if n == -1 else n
    corrector = UnigramCorrector(n)
    print("%i words" % len(corrector.holder))
    print("%i bigrams" % len(corrector.bigrams))

    if parameters["benchmark"] == "0":
        sequences = interactive_sequence_generator()
        writer = None
    else:
        subset = Subset.TEST if parameters["test"] else Subset.DEVELOPMENT
        benchmark = Benchmark(parameters["benchmark"], subset)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        writer = PredictionsFileWriter(benchmark.get_results_directory() + "unigram.txt")

    for s_i, sequence in enumerate(sequences):
        start_time = timestamp()
        predicted = corrector.correct(sequence)
        runtime = time_diff(start_time)
        if writer is None:
            print(predicted)
        else:
            writer.add(predicted, runtime)
    
    if writer is not None:
        writer.save()
