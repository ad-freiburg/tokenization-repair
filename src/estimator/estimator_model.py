import os
import abc
import tensorflow as tf

from src.helper.pickle import load_object, dump_object
from src.settings import paths
from src.helper.files import make_directory, get_files, file_exists, remove_dir, path_exists
from src.tensorflow.fixed_estimator import Estimator
from src.tensorflow.metrics import accuracy
from src.tensorflow.functions import masked_mean
from src.encoding.character_encoder import get_encoder
from src.estimator.training_result_hook import TrainingResultHook
from src.estimator.training_result_saver import TrainingResultSaver


def latest_saved_model_dir(path):
    """
    Finds the most recent timestamped subfolder at the given path.
    Assumes folders are named by a timestamp.
    :param path: path containing at least one timestamped subfolder
    :return: string consisting of path/latest_timestamped_subfolder, or None if no saved model exists
    """
    files = get_files(path)
    saved_model_dirs = [f for f in files if "temp" not in f and file_exists(path + "/" + f + "/saved_model.pb")]
    if len(saved_model_dirs) == 0:
        return None
    latest = sorted([int(model_dir) for model_dir in saved_model_dirs])[-1]
    return path + "/" + str(latest)


def log_metrics(loss, logits, predictions, labels, padding_mask, learning_rate):
    batch_accuracy = accuracy(logits, labels, padding_mask=padding_mask)
    total_accuracy = tf.metrics.accuracy(labels, predictions, weights=padding_mask)[1]
    accuracy_logging_hook = tf.train.LoggingTensorHook({"batch_accuracy": batch_accuracy,
                                                        "total_accuracy": total_accuracy},
                                                       every_n_iter=1)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("batch_accuracy", batch_accuracy)
    tf.summary.scalar("total_accuracy", total_accuracy)
    tf.summary.scalar("learning_rate", learning_rate)
    return accuracy_logging_hook


def masked_loss(logits, labels, padding_mask):
    elementwise_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits)
    loss = masked_mean(elementwise_loss, padding_mask)
    return loss


class EstimatorModel:
    """
    Wrapper around tensorflow's Estimator.
    Functionalities:
    - model creation
    - training
    - save model
    - restore
    - stepwise prediction

    Child classes have to provide functions:
    - model_function: defines the graph and returns EstimatorSpec for training and prediction
    - serving_input_receiver_function: defines the input placeholders
    - _input_dict: creates input tensors from a sequence for prediction
    - _output_dict: collects output tensors from a prediction result

    Training procedure:
    0) call constructor defining #GPUs
    1) initialize: provide a specification that specifies the graph
    2) train: calls the estimator training loop. Give an input function provider.
        input function has to provide a dataset with elements feature_dict and labels
    3) export model: saves the model parameters and placeholders as specified in the serving_input_function to disk.

    For stepwise usage provide functions:
    - initial_state: returning a state dictionary
    - step: returning next state, given current state and input label
    """

    def __init__(self,
                 num_gpus=1,
                 save_checkpoints=True,
                 keep_checkpoints=1,
                 keep_checkpoint_every_hours=1):
        self.specification = None
        self.estimator = None
        self.encoder = None
        self.predict_fn = None
        self.num_gpus = num_gpus
        self.save_checkpoints = save_checkpoints
        self.keep_checkpoints = keep_checkpoints
        self.keep_checkpoint_every_hours = keep_checkpoint_every_hours
        self.training_results = {}

    def initialize(self, specification, encoder):
        """
        Initializes a new estimator model.
        Binds the given specification and encoder to self.
        Removes estimator model with the same name if stored at the default model directory as given by method model_dir.
        Creates Estimator and binds it to self.
        :param specification: specification holding the model's hyperparameters
        :param encoder: encoder that encodes text to subword unit labels (and decodes too)
        """
        self.specification = specification
        self.encoder = encoder
        if path_exists(self.model_dir()):
            remove_dir(self.model_dir())
        self.estimator = self._make_estimator()
        self._save_specification()
        self._save_encoder()

    @abc.abstractmethod
    def model_function(self, features, labels, mode, params):
        """
        Implement in child classes.
        Creates the training and prediction graphs.
        :param features: features as given by training data function or serving input function
        :param labels: target labels for training
        :param mode: training or prediction
        :param params: specification holding model hyperparameters
        :return: EstimatorSpec for training or prediction
        """
        raise NotImplementedError()

    def _make_estimator(self):
        """
        Wraps the model function, and a config including a MirroredStrategy for distributed training
        if number of GPUs > 1 to an Estimator object.
        :return: the model as an Estimator object
        """
        if self.num_gpus > 1:
            distribution_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=self.num_gpus)
        else:
            distribution_strategy = None
        if not self.save_checkpoints:
            config = tf.estimator.RunConfig(log_step_count_steps=1,
                                            keep_checkpoint_max=1,
                                            train_distribute=distribution_strategy,
                                            save_checkpoints_steps=None,
                                            save_checkpoints_secs=None)
        else:
            config = tf.estimator.RunConfig(log_step_count_steps=1,
                                            keep_checkpoint_max=self.keep_checkpoints,
                                            keep_checkpoint_every_n_hours=self.keep_checkpoint_every_hours,
                                            train_distribute=distribution_strategy)
        return Estimator(
            model_fn=self.model_function,
            config=config,
            params=self.specification,
            model_dir=self.model_dir()
        )

    def model_dir(self):
        """
        Returns the model directory composed of default_estimator_folder/model_name/.
        :return: directory as string
        """
        return paths.ESTIMATORS_DIR + self.specification.name + "/"

    def _path_to_file(self, file_name: str) -> str:
        return self.model_dir() + file_name

    def _save_specification(self):
        """
        Stores the specification at the model directory.
        File name is specification.pkl.
        :return:
        """
        make_directory(self.model_dir())
        dump_object(self.specification, self._path_to_file("specification.pkl"))

    def _save_encoder(self):
        """
        Stores the encoder at the model directory.
        File name is encoder.pkl.
        :return:
        """
        make_directory(self.model_dir())
        dump_object(self.encoder, self._path_to_file("encoder.pkl"))

    def save_training_results(self):
        dump_object(self.training_results, self._path_to_file("training_results.pkl"))

    def _load_training_results(self):
        path = self._path_to_file("training_results.pkl")
        if file_exists(path):
            self.training_results = load_object(path)

    def load(self, name):
        """
        Loads an exported model into the default session.
        Assumes the model is stored at paths.ESTIMATOR_FOLDER + name including specification and encoder.
        :param name: name of the model, defines model subfolder
        """
        model_dir = paths.ESTIMATORS_DIR + name + "/"
        self.specification = load_object(model_dir + "specification.pkl")
        try:
            self.encoder = load_object(model_dir + "encoder.pkl")
        except ImportError:
            print("Warning: ImportError when loading encoder.pkl. Recreating encoder instead.")
            self.encoder = get_encoder()
        if self.encoder is None:
            self.encoder = get_encoder()
        self.estimator = self._make_estimator()
        self._update_saved_model()
        self.predict_fn = tf.contrib.predictor.from_saved_model(latest_saved_model_dir(self.model_dir()))

    @abc.abstractmethod
    def _input_dict(self, encoded_sequence):
        """
        Implement in child classes.
        Creates input tensors from a sequence for prediction.
        :param encoded_sequence: list of labels
        :return: dictionary of input tensors that can be fed into predict_fn
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _output_dict(self, encoded_sequence, predict_fn_result):
        """
        Implement in child classes.
        Collects output tensors from a prediction result.
        :param encoded_sequence: list of labels
        :param predict_fn_result: result of the predict_fn
        :return: dictionary of output tensors
        """
        raise NotImplementedError()

    def predict(self, sequence):
        """
        Encode the sequence and predict the next label at each position in the encoded sequence,
        except for the last one (after <endofsequence>).
        Define input_length = len(encoded_sequence) - 1
        :param sequence: input string
        :return: dictionary with entries:
            - "ground_truth": list of ground truth labels, length equals input_length.
            - "predictions": list of predicted labels, length equals input_length.
            - "probabilities": matrix of probabilities, shape equals (input_length, vocabulary_size).
        """
        encoded = self.encoder.encode_sequence(sequence)
        input_dict = self._input_dict(encoded)
        result = self.predict_fn(input_dict=input_dict)
        output_dict = self._output_dict(encoded, result)
        return output_dict
    
    def export_model(self):
        """
        Saves the model as a SavedModel to the default model directory as given by method model_dir.
        Input placeholders are defined by method serving_input_receiver_function.
        """
        self.estimator.export_saved_model(self.model_dir(), self.serving_input_receiver_function)

    @abc.abstractmethod
    def serving_input_receiver_function(self):
        """
        Implement in child classes.
        Must return ServingInputReceiver with placeholders for the input features as used in the model_function.
        Used in method export_model(...) to store the placeholders to disk.
        :return: ServingInputReceiver object
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _data_fn_provider(self, provider_class, encoder, batch_size, max_seq_len, start_bach):
        raise NotImplementedError()

    def train(self, data_function_provider_class, encoder, batch_size, max_seq_len, steps, start_batch):
        data_fn_provider = self.get_data_fn_provider(data_function_provider_class, encoder, batch_size, max_seq_len,
                                                     start_batch)
        self.train_with_data_fn_provider(data_fn_provider, steps)

    def get_data_fn_provider(self, data_function_provider_class, encoder, batch_size, max_seq_len, start_batch):
        return self._data_fn_provider(data_function_provider_class, encoder, batch_size, max_seq_len, start_batch)

    def train_with_data_fn_provider(self, data_fn_provider, steps):
        self._load_training_results()
        self.estimator.train(input_fn=data_fn_provider.train_input_function,
                             steps=steps,
                             saving_listeners=[TrainingResultSaver(self)])

    def _trained_since_last_export(self):
        export_path = latest_saved_model_dir(self.model_dir())
        if export_path is None:
            return True
        export_time = int(export_path.split("/")[-1])
        checkpoint_time = int(os.path.getmtime(self.model_dir() + "/checkpoint"))
        return checkpoint_time > export_time

    def _remove_saved_models(self):
        files = get_files(self.model_dir())
        saved_models = [f for f in files if file_exists(self.model_dir() + "/" + f + "/saved_model.pb")]
        for model_name in saved_models:
            remove_dir(self.model_dir() + "/" + model_name)

    def _update_saved_model(self):
        if self._trained_since_last_export():
            self._remove_saved_models()
            self.export_model()

    @abc.abstractmethod
    def _initial_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, state, x):
        raise NotImplementedError()

    def training_result_hook(self,
                             tensor_name: str,
                             tensor: tf.Tensor):
        return TrainingResultHook(training_result_dict=self.training_results,
                                  tensor_name=tensor_name,
                                  tensor=tensor)
