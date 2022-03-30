import argparse
import tensorflow as tf

import project
from src.encoding.character_encoder import get_encoder
from src.data_fn.file_reader_robust_data_fn_provider import FileReaderRobustDataFnProvider
from src.estimator.bidirectional_labeling_estimator import BidirectionalLabelingEstimator, \
    BidirectionalLabelingEstimatorSpecification


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", dest="model_name", type=str)
    parser.add_argument("-chars", dest="char_frequencies_file", type=str)
    parser.add_argument("-v", dest="voc_size", type=int)
    parser.add_argument("-ru", dest="recurrent_units", type=int, nargs="+")
    parser.add_argument("-du", dest="dense_units", type=int, nargs="+")
    parser.add_argument("-data", dest="dataset", type=str)
    parser.add_argument("-e", dest="epochs", type=int)
    parser.add_argument("-bs", dest="batch_size", type=int)
    parser.add_argument("-sl", dest="sequence_length", type=int)
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    recurrent_units = args.recurrent_units
    if not isinstance(recurrent_units, list):
        recurrent_units = [recurrent_units]
    dense_units = args.dense_units
    if not isinstance(dense_units, list):
        dense_units = [dense_units]

    encoder = get_encoder(args.voc_size, char_frequency_dict_path=args.char_frequencies_file)

    model = BidirectionalLabelingEstimator()

    spec = BidirectionalLabelingEstimatorSpecification(recurrent_units=recurrent_units,
                                                       dense_units=dense_units,
                                                       dim=encoder.dim(),
                                                       name=args.model_name)

    model.initialize(spec, encoder)

    for e_i in range(args.epochs):
        provider = FileReaderRobustDataFnProvider(encoder, batch_size=args.batch_size, max_len=args.sequence_length,
                                                  labeling_output=True, start_batch=0, training_file_path=args.dataset,
                                                  noise_inducer=None)

        model.train_with_data_fn_provider(provider, None)
