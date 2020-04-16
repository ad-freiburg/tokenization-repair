import os

from constants import (
    BACKWARD, BATCH_SIZE, DEFAULT_DATASET,
    DEFAULT_DATA_LOAD_DIR, DEFAULT_MODEL_DUMP_DIR, DEFAULT_MODEL_LOAD_DIR,
    DEFAULT_BENCHMARK,
    DEFAULT_ROOT_DIR, DEFAULT_BENCHMARK_DUMP_DIR, E2E_MODES_ENUM,
    FORWARD, HISTORY_LENGTH, MODELS_ENUM, FIXERS_ENUM, DATASETS_ENUM,
)


class Config:
    def __init__(self, **kwargs):
        self.merge(kwargs)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def merge(self, kwargs):
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    def copy(self):
        return Config(**self.__dict__)


# ############################################### Datasets


def get_dataset_config(dataset=DEFAULT_DATASET, benchmark=DEFAULT_BENCHMARK, **kwargs):
    root_path = DEFAULT_DATA_LOAD_DIR
    benchmarks_root_path = os.path.join(root_path, 'tokenization-repair-paper', 'benchmarks_33')
    return Config(
        dataset=dataset,
        root_path=root_path,
        benchmarks_root_path = os.path.join(root_path, 'tokenization-repair-paper', 'benchmarks_33'),
        benchmark=benchmark,
        train_path=os.path.join(root_path, dataset, 'training_shuffled.txt'),
        valid_correct_path=os.path.join(benchmarks_root_path, benchmark, 'development', 'correct.txt'),
        valid_corrupt_path=os.path.join(benchmarks_root_path, benchmark, 'development', 'corrupt.txt'),
        test_correct_path=os.path.join(benchmarks_root_path, benchmark, 'test', 'correct.txt'),
        test_corrupt_path=os.path.join(benchmarks_root_path, benchmark, 'test', 'corrupt.txt'),
        vocab_path=os.path.join(DEFAULT_ROOT_DIR, 'vocab.txt'),
        dictionary_path=os.path.join(DEFAULT_ROOT_DIR, 'dictionary.pkl'),
    )


# ############################################### Models

def get_language_model_config(
        model=MODELS_ENUM.forward_language_model,
        history_length=HISTORY_LENGTH,
        perturbate=1,
        return_sequences=True,
        rnn_type='LSTM',
        rnn_layers=1,
        rnn_units=1024,
        fully_connected_layers=1,
        fully_connected_units=1024,
        learning_rate=0.01,
        decay_rate=1e-5,
        batch_size=BATCH_SIZE,
        inference=False,
        epochs=100,
        dropout_rate=0.5, **kwargs):
    dataset_config = get_dataset_config(**kwargs)
    direction = {MODELS_ENUM.forward_language_model: FORWARD,
                 MODELS_ENUM.backward_language_model: BACKWARD}.get(model, 'bidir')
    model_repr = '%s-%s-%s-%d_L%d_F%d_%du_dr-%.2f_P%d' % (
        dataset_config.dataset, model, rnn_type, rnn_units, rnn_layers,
        fully_connected_layers, fully_connected_units, dropout_rate, perturbate)

    model_load_path = os.path.join(
        DEFAULT_MODEL_LOAD_DIR, model, model_repr) + '.h5'
    model_save_path = os.path.join(
        DEFAULT_MODEL_DUMP_DIR, model, model_repr) + '.h5'

    config = Config(model_name=model,
                    direction=direction,
                    history_length=history_length,
                    perturbate=perturbate,
                    return_sequences=return_sequences,
                    rnn_type=rnn_type,
                    rnn_layers=rnn_layers,
                    rnn_units=rnn_units,
                    fully_connected_layers=fully_connected_layers,
                    fully_connected_units=fully_connected_units,
                    dropout_rate=dropout_rate,
                    model_repr=model_repr,
                    model_save_path=model_save_path,
                    model_load_path=model_load_path,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    decay_rate=decay_rate,
                    inference=inference,
                    epochs=epochs,
                    )
    config.merge(dataset_config)
    return config


def get_bicontext_fixer_config(beam_size=5,
                               fix_delimiters_only=True,
                               random_sample_files=False,
                               use_default_weights=False,
                               use_timestamp=False,

                               bidir=True, # both those variables should be equivalent
                               bidirectional_weights=True,
                               lflen=1,

                               use_bias=True,
                               use_kernel=True,
                               use_look_forward=False,

                               **kwargs):
    if bidir:
        fixer = FIXERS_ENUM.bicontext_fixer
    else:
        fixer = FIXERS_ENUM.unidirctional_fixer
    config = get_language_model_config(inference=True, **kwargs)
    config.fixer = fixer
    config.beam_size = beam_size
    config.use_look_forward = use_look_forward
    config.fix_delimiters_only = fix_delimiters_only
    config.random_sample_files = random_sample_files
    config.use_default_weights = use_default_weights
    config.use_timestamp = use_timestamp
    config.lflen = lflen
    config.bidir = bidir
    config.bidirectional_weights = bidirectional_weights
    config.use_bias = use_bias
    config.use_kernel = use_kernel
    config.fixer_repr = '%s-%s-%s-%s_%d_L%d_F%d_%du_dr-%.2f_P%d_beam-%d_LF-%d_%s_W%d_f%d_act%d_k%d_b%d' % (
        fixer, config.dataset, config.benchmark, config.rnn_type, config.rnn_units,
        config.rnn_layers, config.fully_connected_layers,
        config.fully_connected_units, config.dropout_rate,
        config.perturbate, beam_size, int(use_look_forward),
        'T' if fix_delimiters_only else 'X', int(use_default_weights),
        lflen, int(bidirectional_weights), int(use_kernel), int(use_bias)
        )
    config.tuner_repr = '%s_%s-%s_%s-%d_L%d_F%d_%du_dr-%.2f_P%d_D%d_f%d_act%d_k%d_b%d' % (
        fixer, config.dataset, config.benchmark, config.rnn_type, config.rnn_units,
        config.rnn_layers, config.fully_connected_layers,
        config.fully_connected_units, config.dropout_rate,
        config.perturbate, int(fix_delimiters_only), lflen,
        int(bidirectional_weights), int(use_kernel), int(use_bias)
        )
    config.tuner_dir = os.path.join(
        DEFAULT_MODEL_LOAD_DIR, 'tuner', config.tuner_repr)
    config.dump_dir = os.path.join(DEFAULT_BENCHMARK_DUMP_DIR, config.fixer_repr)
    print(config.tuner_dir, config.dump_dir)
    return config


def get_dp_config(alpha=1.15, beta=0.1, gamma=1, zeta=2, damping_factor=0.5,
                  use_timestamp=False,
                  window_siz=5, random_sample_files=False, **kwargs):
    config = get_dataset_config(**kwargs)
    config.fixer = FIXERS_ENUM.dp_fixer
    config.alpha = alpha
    config.beta = beta
    config.gamma = gamma
    config.zeta = zeta
    config.damping_factor = damping_factor
    config.use_timestamp = use_timestamp
    config.window_siz = window_siz
    config.random_sample_files = random_sample_files
    config.fixer_repr = '%s-%s-%s-a%.2f_b%.2f_g%.2f_z%.2f_d%.2f_w%d' % (
        config.fixer, config.dataset, config.benchmark, config.alpha,
        config.beta, config.gamma, config.zeta,
        config.damping_factor, config.window_siz)
    config.dump_dir = os.path.join(DEFAULT_BENCHMARK_DUMP_DIR, config.fixer_repr)
    return config


def get_fixer_config(fixer=FIXERS_ENUM.bicontext_fixer, **kwargs):
    if fixer == FIXERS_ENUM.dp_fixer:
        return get_dp_config(**kwargs)
    else:
        return get_bicontext_fixer_config(**kwargs)
