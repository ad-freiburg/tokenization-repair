import argparse
import tensorflow as tf

import project
from src.estimator.unidirectional_lm_estimator import UnidirectionalLMEstimator, UnidirectionalLMEstimatorSpecification
from src.data_fn.file_reader_data_fn_provider import FileReaderDataFnProvider
from src.encoding.character_encoder import get_encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", dest="model_name", type=str, required=True)
    parser.add_argument("-chars", dest="char_frequencies_file", type=str,
                        default="/external/dictionaries/character_frequencies.pkl")
    parser.add_argument("-v", dest="voc_size", type=int, default=200)
    parser.add_argument("-ru", dest="recurrent_units", type=int, nargs="+", default=[1024])
    parser.add_argument("-du", dest="dense_units", type=int, nargs="+", default=[1024])
    parser.add_argument("-data", dest="dataset", type=str, required=True)
    parser.add_argument("-e", dest="epochs", type=int, default=1)
    parser.add_argument("-bs", dest="batch_size", type=int, default=128)
    parser.add_argument("-sl", dest="sequence_length", type=int, default=256)

    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.INFO)

    recurrent_units = args.recurrent_units
    if not isinstance(recurrent_units, list):
        recurrent_units = [recurrent_units]
    dense_units = args.dense_units
    if not isinstance(dense_units, list):
        dense_units = [dense_units]

    model = UnidirectionalLMEstimator(keep_checkpoints=1,
                                      keep_checkpoint_every_hours=9999)

    char_frequencies_file = args.char_frequencies_file
    voc_size = args.voc_size
    encoder = get_encoder(voc_size, char_frequency_dict_path=char_frequencies_file)
    print("Loaded char encoder.")

    spec = UnidirectionalLMEstimatorSpecification(
        backward=False,
        name=args.model_name,
        recurrent_units=recurrent_units,
        dense_units=dense_units,
        dim=encoder.dim(),
        embedding=False,
        embedding_dim=0
    )
    model.initialize(spec, encoder)
    print("Initialised model.")

    for e_i in range(args.epochs):
        provider = FileReaderDataFnProvider(encoder,
                                            batch_size=args.batch_size,
                                            start_batch=0,
                                            pad_sos=False,
                                            max_len=args.sequence_length,
                                            noise_inducer=None,
                                            mask_noisy=False,
                                            bidirectional_mask=False,
                                            dataset_file_path=args.dataset)
        print("Initialised data fn provider.")
        model.train_with_data_fn_provider(provider, steps=None)
