import project
from src.load.load_char_lm import load_char_lm


if __name__ == "__main__":
    model = load_char_lm(model_type="fwd", model_name="fwd1024")
    print(model)
    print(model.model)
    estimator = model.model.estimator
    print(estimator.latest_checkpoint())
    var_names = estimator.get_variable_names()
    for name in var_names:
        value = estimator.get_variable_value(name)
        print(name)
        print(value.shape)
        print(value)
