from typing import Dict, List

import tensorflow as tf


class TrainingResultHook(tf.train.SessionRunHook):
    def __init__(self,
                 training_result_dict: Dict[str, List],
                 tensor_name: str,
                 tensor: tf.Tensor):
        self.training_result_dict = training_result_dict
        self.tensor_name = tensor_name
        self.tensor = tensor

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(fetches=self.tensor)

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values: tf.train.SessionRunValues):
        if self.tensor_name not in self.training_result_dict:
            self.training_result_dict[self.tensor_name] = []
        tensor_result = run_values.results
        self.training_result_dict[self.tensor_name].append(tensor_result)
