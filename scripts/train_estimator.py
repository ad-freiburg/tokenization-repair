from project import src

from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("model_name", "-name", "str"),
    Parameter("direction", "-dir", "str",
              help_message="Choose from {fwd, bwd, bidir}.",
              dependencies=[("bidir", Parameter("sigmoidal", "-sigmoid", "boolean"))]),
    Parameter("vocabulary_size", "-voc", "int"),
    Parameter("recurrent_units", "-ru", "int"),
    Parameter("dense_units", "-du", "int"),
    Parameter("dataset", "-data", "str"),
    Parameter("batch_size", "-bs", "int"),
    Parameter("sequence_length", "-len", "int"),
    Parameter("noise_prob", "-p", "float"),
    Parameter("start_batch", "-start", "int"),
    Parameter("steps", "-steps", "int"),
    Parameter("keep_n_checkpoints", "-keep", "int"),
    Parameter("keep_every_hours", "-every", "int")
]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import tensorflow as tf

from src.estimator.bidirectional_lm_estimator import BidirectionaLLMEstimator, BidirectionalLMEstimatorSpecification
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator, UnidirectionalLMEstimatorSpecification
from src.data_fn.wiki_data_fn_provider import WikiDataFnProvider
from src.data_fn.acl_data_fn_provider import ACLDataFnProvider
from src.encoding.character_encoder import get_encoder


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    recurrent_units = parameters["recurrent_units"]
    if not isinstance(recurrent_units, list):
        recurrent_units = [recurrent_units]
    dense_units = parameters["dense_units"]
    if not isinstance(dense_units, list):
        dense_units = [dense_units]

    bidir = parameters["direction"] == "bidir"
    backward = parameters["direction"] == "bwd"

    keep_checkpoints = parameters["keep_n_checkpoints"]
    keep_every_hours = parameters["keep_every_hours"]

    if bidir:
        model = BidirectionaLLMEstimator(keep_checkpoints=keep_checkpoints,
                                         keep_checkpoint_every_hours=keep_every_hours)
    else:
        model = UnidirectionalLMEstimator(keep_checkpoints=keep_checkpoints,
                                          keep_checkpoint_every_hours=keep_every_hours)

    if parameters["start_batch"] == 0:
        encoder = get_encoder(parameters["vocabulary_size"])
        print("Loaded char encoder.")
        if bidir:
            spec = BidirectionalLMEstimatorSpecification(
                name=parameters["model_name"],
                recurrent_units=recurrent_units,
                dense_units=dense_units,
                dim=encoder.dim(),
                sigmoidal=parameters["sigmoidal"],
                embedding=False,
                embedding_dim=0
            )
        else:
            spec = UnidirectionalLMEstimatorSpecification(
                backward=backward,
                name=parameters["model_name"],
                recurrent_units=recurrent_units,
                dense_units=dense_units,
                dim=encoder.dim(),
                embedding=False,
                embedding_dim=0
            )
        model.initialize(spec, encoder)
        print("Initialised model.")
    else:
        model.load(parameters["model_name"])
        encoder = model.encoder
        print("Loaded model.")
        if parameters["dataset"] == "acl" and not model.specification.name.endswith("acl"):
            model.specification.name = model.specification.name + "_acl"
            print("renamed model to %s" % model.specification.name)
            model._save_specification()
            model._save_encoder()

    p = parameters["noise_prob"]
    if p == 0:
        p = None

    if parameters["dataset"] == "acl":
        provider_class = ACLDataFnProvider
    else:
        provider_class = WikiDataFnProvider
    provider = provider_class(encoder,
                              batch_size=parameters["batch_size"],
                              start_batch=parameters["start_batch"],
                              pad_sos=backward,
                              max_len=parameters["sequence_length"],
                              noise_prob=p,
                              mask_noisy=bidir,
                              seed=42,
                              bidirectional_mask=bidir)
    print("Initialised data fn provider.")

    steps = parameters["steps"]
    remaining_steps = None if steps == -1 else (steps - parameters["start_batch"])

    model.train_with_data_fn_provider(provider, remaining_steps)
