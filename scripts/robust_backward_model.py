import project
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator
from src.interactive.sequence_generator import interactive_sequence_generator
from src.helper.data_structures import top_k_indices_sorted


if __name__ == "__main__":
    name = "robust_bwd_model"
    model = UnidirectionalLMEstimator()
    model.load(name)
    space_label = model.encoder.encode_char(' ')
    for sequence in interactive_sequence_generator():
        result = model.predict(sequence)
        probabilities = result["probabilities"]
        print(probabilities.shape)
        for char, probs in zip(sequence, probabilities[:-1]):
            top = top_k_indices_sorted(probs, 5)
            print_str = "%s %.4f |" % (char, probs[space_label])
            for label in top:
                print_str += " %s %.4f" % (model.encoder.decode_label(label), probs[label])
            print(print_str)
