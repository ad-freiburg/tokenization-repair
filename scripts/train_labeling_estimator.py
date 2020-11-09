import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model_name", "-name", "str"),
          Parameter("vocabulary", "-voc", "str"),
          Parameter("dataset", "-data", "str"),
          Parameter("noise", "-noise", "float"),
          Parameter("batch_size", "-bs", "int"),
          Parameter("epochs", "-e", "int"),
          Parameter("start_batch", "-start", "int")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import tensorflow as tf

from src.encoding.character_encoder import get_encoder, get_arxiv_encoder
from src.data_fn.robust_data_fn_provicer import RobustDataFnProvider
from src.data_fn.acl_robust_data_fn_provider import ACLRobustDataFnProvider
from src.data_fn.arxiv_robust_data_fn_provider import ArxivRobustDataFnProvider
from src.data_fn.file_reader_robust_data_fn_provider import FileReaderRobustDataFnProvider
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator, \
    BidirectionalLabelingEstimatorSpecification


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
        if parameters["dataset"] == "acl" and not model.specification.name.endswith("acl"):
            model.specification.name = model.specification.name + "_acl"
            print("renamed model to %s" % model.specification.name)
            model._save_specification()
            model._save_encoder()

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

        provider = provider_class(encoder, batch_size=batch_size, noise_prob=noise, max_len=seq_len, seed=42,
                                  labeling_output=True, start_batch=start_batch, training_file_path=training_file_path)

        model.train_with_data_fn_provider(provider, None)
