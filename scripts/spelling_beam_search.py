import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("benchmark", "-benchmark", "str"),
          Parameter("n_sequences", "-n", "int"),
          Parameter("n_beams", "-b", "int"),
          Parameter("space_penalty", "-sp", "float"),
          Parameter("char_penalty", "-cp", "float"),
          Parameter("out_file", "-f", "str"),
          Parameter("segmentation_file", "-seg", "str"),
          Parameter("continue", "-c", "boolean")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.interactive.sequence_generator import interactive_sequence_generator
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.spelling.spelling_beam_search_corrector import SpellingBeamSearchCorrector
from src.benchmark.benchmark import BenchmarkFiles, Subset, Benchmark
from src.evaluation.predictions_file_writer import PredictionsFileWriter
from src.helper.time import time_diff, timestamp


if __name__ == "__main__":
    model = UnidirectionalLMEstimator()
    model.load("fwd1024")
    corrector = SpellingBeamSearchCorrector(model,
                                            n_beams=parameters["n_beams"],
                                            branching_factor=parameters["n_beams"],
                                            consecutive_insertions=2,
                                            char_penalty=parameters["char_penalty"],
                                            space_penalty=parameters["space_penalty"])

    benchmark_name = parameters["benchmark"]
    if benchmark_name != "0":
        benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
        sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
        n_sequences = parameters["n_sequences"]
        file_writer = PredictionsFileWriter(benchmark.get_results_directory() + parameters["out_file"])
        segmentation_file_writer = PredictionsFileWriter(benchmark.get_results_directory() +
                                                         parameters["segmentation_file"])
        if parameters["continue"]:
            file_writer.load()
            segmentation_file_writer.load()
        corrector.verbose = False
    else:
        sequences = interactive_sequence_generator()
        n_sequences = -1
        file_writer = None
        segmentation_file_writer = None

    for s_i, sequence in enumerate(sequences):
        if s_i == n_sequences:
            break
        if file_writer is not None:
            if s_i < file_writer.n_sequences():
                continue
            print("sequence %i" % s_i)
            print(sequence)
        start_time = timestamp()
        predicted, segmentation = corrector.correct(sequence)
        runtime = time_diff(start_time)
        print(predicted)
        print(segmentation)
        if file_writer is not None:
            file_writer.add(predicted, runtime)
            file_writer.save()
            segmentation_file_writer.add(segmentation, runtime)
            segmentation_file_writer.save()
        else:
            print(runtime)
            print(corrector.total_model_time)
