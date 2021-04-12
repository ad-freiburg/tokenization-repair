import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model_name", "-name", "str"),
          Parameter("vocabulary", "-voc", "str"),
          Parameter("dataset", "-data", "str"),
          Parameter("noise", "-noise", "str"),
          Parameter("batch_size", "-bs", "int"),
          Parameter("epochs", "-e", "int"),
          Parameter("start_batch", "-start", "int")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import tensorflow as tf

from src.encoding.character_encoder import get_encoder, get_arxiv_encoder, get_mixed_encoder
from src.data_fn.robust_data_fn_provicer import RobustDataFnProvider
from src.data_fn.acl_robust_data_fn_provider import ACLRobustDataFnProvider
from src.data_fn.arxiv_robust_data_fn_provider import ArxivRobustDataFnProvider
from src.data_fn.file_reader_robust_data_fn_provider import FileReaderRobustDataFnProvider
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator, \
    BidirectionalLabelingEstimatorSpecification
from src.noise.token_typo_inducer import TokenTypoInducer
from src.noise.ocr_noise_inducer import OCRNoiseInducer
from src.noise.char_and_punctuation_noise_inducer import CharAndPunctuationNoiseInducer
from src.noise.acl_noise_inducer import ACLNoiseInducer


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    name = parameters["model_name"]

    recurrent_units = [1024]
    dense_units = [1024]
    seq_len = 256
    batch_size = parameters["batch_size"]
    noise = parameters["noise"]
    start_batch = parameters["start_batch"]

    if parameters["vocabulary"] == "arxiv":
        encoder = get_arxiv_encoder()
    elif parameters["vocabulary"] == "mixed":
        encoder = get_mixed_encoder()
    else:
        vocab_size = int(parameters["vocabulary"])
        encoder = get_encoder(vocab_size)

    model = BidirectionalLabelingEstimator()

    if start_batch == 0:
        spec = BidirectionalLabelingEstimatorSpecification(recurrent_units=recurrent_units,
                                                           dense_units=dense_units,
                                                           dim=encoder.dim(),
                                                           name=name)

        model.initialize(spec, encoder)
    else:
        model.load(name)
        print("Model loaded.")

    noise_inducer = None
    if parameters["noise"] == "ocr":
        noise_inducer = OCRNoiseInducer(p=0.05, seed=1337)
    elif parameters["noise"] == "new":
        noise_inducer = CharAndPunctuationNoiseInducer(p=0.2, seed=1337)
    elif parameters["noise"] == "acl":
        noise_inducer = ACLNoiseInducer(p=0.1, insertion_prob=0.2079, seed=1337)
    else:
        p_noise = float(parameters["noise"])
        if p_noise > 0:
            noise_inducer = TokenTypoInducer(p_noise, seed=1337)

    for e_i in range(parameters["epochs"]):
        training_file_path = None
        if parameters["dataset"] == "acl":
            provider_class = ACLRobustDataFnProvider
        elif parameters["dataset"] == "arxiv":
            provider_class = ArxivRobustDataFnProvider
        elif parameters["dataset"] == "wikipedia":
            provider_class = RobustDataFnProvider
        else:
            provider_class = FileReaderRobustDataFnProvider
            training_file_path = parameters["dataset"]

        print("training file path:", training_file_path)

        provider = provider_class(encoder, batch_size=batch_size, max_len=seq_len, labeling_output=True,
                                  start_batch=start_batch, training_file_path=training_file_path,
                                  noise_inducer=noise_inducer)

        model.train_with_data_fn_provider(provider, None)
