import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model_name", "-name", "str"),
          Parameter("noise", "-noise", "float"),
          Parameter("batch_size", "-bs", "int"),
          Parameter("start_batch", "-start", "int")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import tensorflow as tf

from src.encoding.character_encoder import get_encoder
from src.data_fn.robust_data_fn_provicer import RobustDataFnProvider
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator, \
    BidirectionalLabelingEstimatorSpecification


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    name = parameters["model_name"]

    vocab_size = 200
    recurrent_units = [1024]
    dense_units = [1024]
    seq_len = 256
    batch_size = parameters["batch_size"]
    noise = parameters["noise"]
    start_batch = parameters["start_batch"]

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

    provider = RobustDataFnProvider(encoder, batch_size=batch_size, noise_prob=noise, max_len=seq_len, seed=42,
                                    labeling_output=True, start_batch=start_batch)

    model.train_with_data_fn_provider(provider, None)
