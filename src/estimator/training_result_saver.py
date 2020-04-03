import tensorflow as tf


class TrainingResultSaver(tf.train.CheckpointSaverListener):
    def __init__(self, model):
        self.model = model

    def after_save(self, session, global_step_value):
        self.model.save_training_results()
        print("saved results for step %i" % global_step_value)
