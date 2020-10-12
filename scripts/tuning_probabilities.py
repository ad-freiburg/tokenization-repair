import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model_name", "-m", "str"),
          Parameter("benchmark", "-b", "str"),
          Parameter("noise", "-noise", "boolean"),
          Parameter("continue", "-c", "boolean")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.helper.pickle import dump_object, load_object
from src.settings import paths
from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles
from src.corrector.beam_search.penalty_tuning import Case


if __name__ == "__main__":
    model_name = parameters["model_name"]
    if parameters["benchmark"] == "acl_all":
        benchmark_name = parameters["benchmark"]
    else:
        benchmark_name = "0.1_inf" if parameters["noise"] else "0_inf"
    path = paths.CASES_FILE_NOISY if parameters["noise"] else paths.CASES_FILE_CLEAN
    path = path % model_name

    LOOKAHEAD = 5

    model = UnidirectionalLMEstimator()
    model.load(model_name)

    space_label = model.encoder.encode_char(' ')

    if parameters["continue"]:
        cases = load_object(path)
    else:
        cases = []

    benchmark = Benchmark(benchmark_name, Subset.TUNING)
    sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)

    for s_i, sequence in enumerate(sequences):
        if s_i < len(cases):
            continue

        print("sequence %i" % s_i)
        print(sequence)
        cases.append([])

        encoded = model.encoder.encode_sequence(sequence)
        if model.specification.backward:
            sequence = sequence[::-1]
            encoded = encoded[::-1]

        state = model.initial_state()

        for i in range(len(sequence) + 1):
            x = encoded[i]
            state = model.step(state, x, include_sequence=False)

            char = sequence[i] if i < len(sequence) else "EOS"
            p_space = state["probabilities"][space_label]
            #print(i, sequence[i - 1] if i > 0 else "SOS", p_space)
            is_space = char == ' '
            next_index = i + (2 if is_space else 1)
            next_labels = encoded[next_index:(next_index + LOOKAHEAD)]
            next_chars = sequence[(next_index - 1):(next_index + LOOKAHEAD - 1)]
            space_state = model.step(state, space_label, include_sequence=False)
            no_space_state = state
            p_after_space = []
            p_after_no_space = []
            for j, label in enumerate(next_labels):
                char = model.encoder.decode_label(label)
                space_p = space_state["probabilities"][label]
                no_space_p = no_space_state["probabilities"][label]
                #print("", j, label, char, space_p, no_space_p)
                p_after_space.append(space_p)
                p_after_no_space.append(no_space_p)
                if j < LOOKAHEAD:
                    space_state = model.step(space_state, label, include_sequence=False)
                    no_space_state = model.step(no_space_state, label, include_sequence=False)
            case = Case(sequence_index=s_i,
                        position=i,
                        true_space=is_space,
                        p_space=p_space,
                        p_after_space=p_after_space,
                        p_after_no_space=p_after_no_space)
            cases[-1].append(case)
        if model.specification.backward:
            cases[-1] = cases[-1][::-1]

        if (s_i + 1) % 1000 == 0:
            dump_object(cases, path)
            print("saved.")
