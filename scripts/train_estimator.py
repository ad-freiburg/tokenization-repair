from project import src

from src.interactive.parameters import Parameter, ParameterGetter


params = [
    Parameter("model_name", "-name", "str"),
    Parameter("direction", "-dir", "str",
              help_message="Choose from {fwd, bwd, bidir}.",
              dependencies=[("bidir", Parameter("sigmoidal", "-sigmoid", "boolean"))]),
    Parameter("vocabulary", "-voc", "str"),
    Parameter("recurrent_units", "-ru", "int"),
    Parameter("dense_units", "-du", "int"),
    Parameter("dataset", "-data", "str"),
    Parameter("epochs", "-e", "int", default=1),
    Parameter("batch_size", "-bs", "int"),
    Parameter("sequence_length", "-len", "int"),
    Parameter("noise", "-noise", "str"),
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
from src.data_fn.acl_corpus_data_fn_provider import ACLCorpusDataFnProvider
from src.data_fn.arxiv_data_fn_provider import ArxivDataFnProvider
from src.data_fn.file_reader_data_fn_provider import FileReaderDataFnProvider
from src.encoding.character_encoder import get_encoder, get_acl_encoder, get_arxiv_encoder
from src.noise.token_typo_inducer import TokenTypoInducer
from src.noise.ocr_noise_inducer import OCRNoiseInducer


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
        if parameters["vocabulary"] == "acl":
            encoder = get_acl_encoder()
        elif parameters["vocabulary"] == "arxiv":
            encoder = get_arxiv_encoder()
        else:
            voc_size = int(parameters["vocabulary"])
            encoder = get_encoder(voc_size)
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
            model.rename(model.specification.name + "_acl")
            print("renamed model to %s" % model.specification.name)

    if parameters["noise"] == "ocr":
        noise_inducer = OCRNoiseInducer(p=0.05, seed=42)
    else:
        p = float(parameters["noise"])
        if p > 0:
            noise_inducer = TokenTypoInducer(p, seed=42)
        else:
            noise_inducer = None

    dataset_file_path = None
    if parameters["dataset"] == "acl":
        provider_class = ACLDataFnProvider
    elif parameters["dataset"] == "acl-all":
        provider_class = ACLCorpusDataFnProvider
    elif parameters["dataset"] == "arxiv":
        provider_class = ArxivDataFnProvider
    elif parameters["dataset"] == "wikipedia":
        provider_class = WikiDataFnProvider
    else:
        provider_class = FileReaderDataFnProvider
        dataset_file_path = parameters["dataset"]

    for e_i in range(parameters["epochs"]):
        provider = provider_class(encoder,
                                  batch_size=parameters["batch_size"],
                                  start_batch=parameters["start_batch"],
                                  pad_sos=backward,
                                  max_len=parameters["sequence_length"],
                                  noise_inducer=noise_inducer,
                                  mask_noisy=bidir,
                                  bidirectional_mask=bidir,
                                  dataset_file_path=dataset_file_path)
        print("Initialised data fn provider.")

        steps = parameters["steps"]
        remaining_steps = None if steps == -1 else (steps - parameters["start_batch"])

        model.train_with_data_fn_provider(provider, remaining_steps)
