from project import src
from src.interactive.parameters import ParameterGetter, Parameter

params = [
    Parameter("name", "-name", "str"),
    Parameter("corruption_probability", "-p", "float"),
    Parameter("noise_probability", "-noise", "float")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.datasets.wikipedia import Wikipedia
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.noise.typo_noise_inducer import TypoNoiseInducer
from src.sequence.token_corruptor import TokenCorruptor
from src.settings import constants
from src.helper.files import open_file


if __name__ == "__main__":
    SEED = 3010

    p = parameters["corruption_probability"]

    # create empty benchmarks
    benchmark_name = parameters["name"]
    development_benchmark = Benchmark(benchmark_name, Subset.DEVELOPMENT)
    test_benchmark = Benchmark(benchmark_name, Subset.TEST)

    # read sequences
    development_sequences = Wikipedia.development_sequences()
    test_sequences = Wikipedia.test_sequences()

    # typo inserter
    p_noise = parameters["noise_probability"]
    typo_inducer = TypoNoiseInducer(p_noise, SEED) if p_noise > 0 else None

    # token corruptor
    corruptor = TokenCorruptor(p=p,
                               positions_per_token=constants.POSITIONS_PER_TOKEN,
                               token_pairs_per_token=constants.TOKEN_PAIRS_PER_TOKEN,
                               seed=SEED)

    for is_development, benchmark, true_sequences in ((True, development_benchmark, development_sequences),
                                                      (False, test_benchmark, test_sequences)):
        correct_file = open_file(benchmark.get_file(BenchmarkFiles.CORRECT))
        corrupt_file = open_file(benchmark.get_file(BenchmarkFiles.CORRUPT))
        insertions_file = open_file(benchmark.get_file(BenchmarkFiles.INSERTIONS)) if is_development else None
        deletions_file = open_file(benchmark.get_file(BenchmarkFiles.DELETIONS)) if is_development else None
        original_file = open_file(benchmark.get_file(BenchmarkFiles.ORIGINAL)) if p_noise > 0 else None

        for original_sequence in true_sequences:
            correct_sequence = original_sequence

            # typo noise:
            if p_noise > 0:
                original_file.write(original_sequence + '\n')
                correct_sequence = typo_inducer.induce_noise(original_sequence)

            # tokenization errors:
            corrupt_sequence = corruptor.corrupt(correct_sequence)
            correct_file.write(correct_sequence + '\n')
            corrupt_file.write(corrupt_sequence + '\n')

            # sequences for threshold fitting:
            if is_development:
                insertions_sequence = corruptor.corrupt(correct_sequence, delete=False)
                deletions_sequence = corruptor.corrupt(correct_sequence, insert=False)
                insertions_file.write(insertions_sequence + '\n')
                deletions_file.write(deletions_sequence + '\n')

        correct_file.close()
        corrupt_file.close()
        if is_development:
            insertions_file.close()
            deletions_file.close()
        if p_noise > 0:
            original_file.close()
