import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("approach", "-a", "str"),
    Parameter("noise_level", "-n", "float")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from enum import Enum
import numpy as np

from src.load.load_char_lm import load_default_char_lm
from src.benchmark.benchmark import Subset, BenchmarkFiles, get_benchmark
from src.sequence.transformation import space_corruption_positions
from src.optimization.threshold import optimal_f1_threshold
from src.corrector.threshold_holder import ThresholdHolder, FittingMethod, ThresholdType


class OperationType(Enum):
    INSERTION = 0
    DELETION = 1


class PredictionType(Enum):
    FALSE_POSITIVE = 0
    TRUE_POSITIVE = 1


class ProbabilityHolder:
    def __init__(self):
        self.probabilities = {
            op_type: {
                pred_type: [] for pred_type in PredictionType
            } for op_type in OperationType
        }

    def add(self, operation_type: OperationType, prediction_type: PredictionType, probability: float):
        self.probabilities[operation_type][prediction_type].append(probability)


def combined_compared_space_probability(p_fwd, p_bwd, b, a, space_index):
    p_space = p_fwd[space_index] * p_bwd[space_index]
    #print("p_space = %f * % f = %f" % (p_fwd[space_index], p_bwd[space_index], p_space))
    #normalizer = np.sum(p_fwd * p_bwd)
    #print("p_space_normalized = %f / %f = % f" % (p_space, normalizer, p_space / normalizer))
    p_no_space = p_fwd[a] * p_bwd[b]
    #print("p_no_space = %f * %f = %f" % (p_fwd[a], p_bwd[b], p_no_space))
    p_compared = p_space / (p_space + p_no_space)
    #print("p_compared = %f / %f = %f" % (p_space, p_space + p_no_space, p_compared))
    return p_compared


if __name__ == "__main__":
    noise_level = parameters["noise_level"]
    error_probabilities = np.arange(0.1, 1.1, 0.1)

    approach = parameters["approach"]
    is_combined = approach.startswith("combined")
    model = load_default_char_lm(approach)
    space_index = model.get_encoder().encode_char(' ')

    benchmarks = {p: get_benchmark(noise_level, p, Subset.DEVELOPMENT) for p in error_probabilities}
    corrupt_sequences = {p: benchmarks[p].get_sequences(BenchmarkFiles.CORRUPT) for p in error_probabilities}
    probability_holders = {p: ProbabilityHolder() for p in error_probabilities}

    for s_i, correct in enumerate(benchmarks[0.1].get_sequences(BenchmarkFiles.CORRECT)):
        print("sequence %i" % s_i)
        prediction = model.predict(correct)

        for p in error_probabilities:
            corrupt = corrupt_sequences[p][s_i]
            inserted_spaces, deleted_spaces = space_corruption_positions(correct, corrupt)
            for i, char in enumerate(correct):
                if char == ' ':
                    if is_combined:
                        space_prob = combined_compared_space_probability(p_fwd=prediction["forward_probabilities"][i],
                                                                         p_bwd=prediction["backward_probabilities"][i],
                                                                         b=prediction["labels"][i - 1],
                                                                         a=prediction["labels"][i + 1],
                                                                         space_index=space_index)
                    else:
                        space_prob = prediction["probabilities"][i][space_index]
                    if i in deleted_spaces:
                        probability_holders[p].add(OperationType.INSERTION, PredictionType.TRUE_POSITIVE, space_prob)
                    else:
                        del_prob = 1 - space_prob
                        probability_holders[p].add(OperationType.DELETION, PredictionType.FALSE_POSITIVE, del_prob)
                else:
                    space_prob = prediction["insertion_probabilities"][i][space_index]
                    if i in inserted_spaces:
                        del_prob = 1 - space_prob
                        probability_holders[p].add(OperationType.DELETION, PredictionType.TRUE_POSITIVE, del_prob)
                    else:
                        probability_holders[p].add(OperationType.INSERTION, PredictionType.FALSE_POSITIVE, space_prob)

    for p in error_probabilities:
        print(p)

        probability_holder = probability_holders[p]

        print("insertion:")
        insertion_threshold = optimal_f1_threshold(
            probability_holder.probabilities[OperationType.INSERTION][PredictionType.TRUE_POSITIVE],
            probability_holder.probabilities[OperationType.INSERTION][PredictionType.FALSE_POSITIVE]
        )

        print("deletion:")
        deletion_threshold = optimal_f1_threshold(
            probability_holder.probabilities[OperationType.DELETION][PredictionType.TRUE_POSITIVE],
            probability_holder.probabilities[OperationType.DELETION][PredictionType.FALSE_POSITIVE]
        )

        threshold_holder = ThresholdHolder(FittingMethod.SINGLE_RUN, autosave=False)

        for threshold, threshold_type in ((insertion_threshold, ThresholdType.INSERTION_THRESHOLD),
                                          (deletion_threshold, ThresholdType.DELETION_THRESHOLD)):
            threshold_holder.set_threshold(threshold_type=threshold_type,
                                           model_name=parameters["approach"],
                                           threshold=threshold,
                                           noise_type=benchmarks[p].name)

        threshold_holder.save()
