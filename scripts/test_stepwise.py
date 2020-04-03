import project
from src.load.load_char_lm import load_char_lm


if __name__ == "__main__":
    model = load_char_lm(model_type="fwd",
                         model_name="fwd1024")
    print(model)
    sequence = "This is a test."
    result = model.predict(sequence)
    probabilities = result["probabilities"]
    print(len(sequence))
    print(len(result["predictions"]))
    print(probabilities.shape)

    state = model.model.initial_state()
    encoded = model.get_encoder().encode_sequence(sequence)
    for i, x in enumerate(encoded):
        if i > 0:
            symbol = model.get_encoder().decode_label(x)
            print(symbol, state["probabilities"][x])
        state = model.model.step(state, x)
