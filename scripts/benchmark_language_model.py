"""
Evaluates a language model's performance.
"""


from project import src
from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("model_type", "-type", "str",
              help_message="Choose from {fwd, bwd, bidir, combined}.",
              dependencies=[
                  ("combined", [Parameter("bwd_model_name", "-bwd", "str")])
              ]),
    Parameter("model_name", "-name", "str"),
    Parameter("benchmark", "-b", "str",
              help_message="Choose from {new, old}. 'old' is the project dev set and 'new' the paper dev set."),
    Parameter("n_sequences", "-n", "int"),
    Parameter("interactive", "-i", "boolean")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


from src.datasets.wikipedia import Wikipedia
from src.load.load_char_lm import load_char_lm
from src.evaluation.language_model_evaluator import LanguageModelEvaluator
from src.settings import paths


def old_validation_sequences(n):
    file_path = paths.PROJECT_DEVELOPMENT_FILE
    with open(file_path) as f:
        for l_i, line in enumerate(f):
            if l_i == n:
                break
            sequence = line[:-1]
            yield sequence


def new_validation_sequences(n):
    return Wikipedia.development_sequences(n)


if __name__ == "__main__":
    model_type = parameters["model_type"]
    model_name = parameters["model_name"]
    bwd_model_name = parameters["bwd_model_name"] if "bwd_model_name" in parameters else None
    model = load_char_lm(model_type, model_name, bwd_model_name)

    encoder = model.get_encoder()
    evaluator = LanguageModelEvaluator(encoder)
    
    if parameters["benchmark"] == "new":
        wiki_sequences = new_validation_sequences(parameters["n_sequences"])
    elif parameters["benchmark"] == "old":
        wiki_sequences = old_validation_sequences(parameters["n_sequences"])
    
    n_wiki_sequences = 0

    while True:
        if parameters["interactive"]:
            sequence = input("> ")
            if sequence == "exit":
                break
            if sequence == "":
                if n_wiki_sequences == parameters["n_sequences"]:
                    break
                sequence = next(wiki_sequences)
                n_wiki_sequences += 1
        else:
            if n_wiki_sequences == parameters["n_sequences"]:
                break
            sequence = next(wiki_sequences)
            n_wiki_sequences += 1
        print(sequence)

        prediction = model.predict(sequence)

        evaluator.register_result(sequence, prediction)

    evaluator.print_summary()
