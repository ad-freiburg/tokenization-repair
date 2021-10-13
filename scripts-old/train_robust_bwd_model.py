import project
from src.interactive.parameters import Parameter, ParameterGetter


params = [Parameter("model_name", "-name", "str"),
          Parameter("noise", "-noise", "float"),
          Parameter("start_batch", "-start", "int")]
getter = ParameterGetter(params)
getter.print_help()
parameters = getter.get()


import tensorflow as tf

from src.encoding.character_encoder import get_encoder
from src.data_fn.robust_data_fn_provicer import RobustDataFnProvider
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimatorSpecification, UnidirectionalLMEstimator


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    name = parameters["model_name"]

    vocab_size = 200
    recurrent_units = [1024]
    dense_units = [1024]
    seq_len = 256
    batch_size = 128
    noise = parameters["noise"]
    start_batch = parameters["start_batch"]

    encoder = get_encoder(vocab_size)
    model = UnidirectionalLMEstimator()

    if start_batch == 0:
        spec = UnidirectionalLMEstimatorSpecification(backward=True,
                                                      embedding=False,
                                                      embedding_dim=None,
                                                      recurrent_units=recurrent_units,
                                                      dense_units=dense_units,
                                                      dim=encoder.dim(),
                                                      name=name,
                                                      x_and_y_input=True)

        model.initialize(spec, encoder)
    else:
        model.load(name)

    provider = RobustDataFnProvider(encoder, batch_size=batch_size, noise_prob=noise, max_len=seq_len, seed=42,
                                    start_batch=start_batch)

    model.train_with_data_fn_provider(provider, None)
