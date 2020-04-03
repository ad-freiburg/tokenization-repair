import tensorflow as tf
import os
import pickle


class BidirectionalModelSpecification:
    def __init__(self, lstm_units, dense_units, dims, sigmoid=False, name=None):
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dims = dims
        self.sigmoid = sigmoid
        self.name = name


class BidirectionalModel:
    def __init__(self):
        self.specification = None

    def initialize(self, specification):
        self.specification = specification
        self._make_graph()
        self._make_insertion_graph()
        self._make_collections()
        self._initialize_variables()

    def close_session(self):
        self.session.close()

    def _make_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # inputs
            self.batch_size = tf.placeholder(tf.int32, [])
            self.seq_len = tf.placeholder(tf.int32)
            self.x = tf.placeholder(tf.float32, shape=(None, None, self.specification.dims))  # batch_size, seq_len, dim
            self.y = tf.placeholder(tf.int32, shape=(None, None))  # batch_size, seq_len

            # reverse sequence
            x_reverse = self.x[:, ::-1, :]

            # recurrent parts
            with tf.variable_scope("fwd"):
                lstm_cell_fwd = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.BasicLSTMCell(units) for units in self.specification.lstm_units]
                )
                self.initial_fwd_state = lstm_cell_fwd.zero_state(self.batch_size, tf.float32)
                self.fwd_outputs, self.fwd_state = tf.nn.dynamic_rnn(
                    cell=lstm_cell_fwd,
                    inputs=self.x,
                    initial_state=self.initial_fwd_state)

            with tf.variable_scope("bwd"):
                lstm_cell_bwd = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.BasicLSTMCell(units) for units in self.specification.lstm_units]
                )
                self.initial_bwd_state = lstm_cell_bwd.zero_state(self.batch_size, tf.float32)
                self.bwd_outputs, self.bwd_state = tf.nn.dynamic_rnn(
                    cell=lstm_cell_bwd,
                    inputs=x_reverse,
                    initial_state=self.initial_bwd_state
                )

            # stack recurrent outputs
            fwd_sliced = tf.slice(self.fwd_outputs,
                                  [0, 0, 0],
                                  [self.batch_size, self.seq_len - 2, self.specification.lstm_units[-1]])
            bwd_sliced = tf.slice(self.bwd_outputs,
                                  [0, 0, 0],
                                  [self.batch_size, self.seq_len - 2, self.specification.lstm_units[-1]])
            stacked = tf.concat([fwd_sliced,
                                 bwd_sliced[:, ::-1, :]],
                                axis=2)
            stacked = tf.reshape(stacked, [self.batch_size, self.seq_len - 2, 2 * self.specification.lstm_units[-1]])

            # dense part
            with tf.variable_scope("dense"):
                dense = tf.layers.dense(stacked, self.specification.dense_units[0], activation=tf.nn.relu,
                                        name="dense")
                for dense_layer, dense_units in enumerate(self.specification.dense_units[1:]):
                    dense = tf.layers.dense(dense, dense_units, activation=tf.nn.relu, name="dense_%i" % dense_layer)
                activation = tf.layers.dense(dense, self.specification.dims, name="activation")

            # output layer and loss function
            if self.specification.sigmoid:
                self.probabilities = tf.nn.sigmoid(activation)
                y_one_hot = tf.one_hot(self.y, self.specification.dims)
                loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_one_hot[:, 1:-1, :],
                                                                      logits=activation)
                self.loss = tf.reduce_sum(loss_matrix)
            else:
                self.probabilities = tf.nn.softmax(activation)
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y[:, 1:-1],
                                                                   logits=activation)
            # optimizer
            with tf.variable_scope("optimizer"):
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.session = tf.Session(graph=self.graph)

    def _make_insertion_graph(self):
        with self.graph.as_default():
            # dense part:
            fwd_sliced = tf.slice(self.fwd_outputs,
                                  [0, 0, 0],
                                  [self.batch_size, self.seq_len - 1, self.specification.lstm_units[-1]])
            bwd_sliced = tf.slice(self.bwd_outputs,
                                  [0, 0, 0],
                                  [self.batch_size, self.seq_len - 1, self.specification.lstm_units[-1]])
            stacked = tf.concat([fwd_sliced,
                                 bwd_sliced[:, ::-1, :]],
                                axis=2)
            stacked_reshaped = tf.reshape(stacked, [self.batch_size,
                                                    self.seq_len - 1,
                                                    2 * self.specification.lstm_units[-1]])
            with tf.variable_scope("dense", reuse=True):
                dense = tf.layers.dense(stacked, self.specification.dense_units[0], activation=tf.nn.relu,
                                        name="dense")
                for dense_layer, dense_units in enumerate(self.specification.dense_units[1:]):
                    dense = tf.layers.dense(dense, dense_units, activation=tf.nn.relu, name="dense_%i" % dense_layer)
                activation = tf.layers.dense(dense, self.specification.dims, name="activation")
            if self.specification.sigmoid:
                self.insertion_probabilities = tf.sigmoid(activation)
                self.y_insertion = tf.placeholder(tf.float32, shape=(None, None, self.specification.dims))
                positional_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_insertion,
                                                                            logits=activation)
                self.insertion_loss = tf.reduce_sum(positional_losses)
                with tf.variable_scope("optimizer", reuse=True):
                    self.insertion_optimizer = tf.train.AdamOptimizer().minimize(self.insertion_loss)
            else:
                self.insertion_probabilities = tf.nn.softmax(activation)

    def _initialize_variables(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.session.run(init)

    def _save_variables_to_directory(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, path + "model")

    def _save_specification(self, path):
        with open(path + "specification.pkl", "wb") as file:
            pickle.dump(self.specification, file)

    def _load_specification(self, path):
        with open(path + "specification.pkl", "rb") as file:
            self.specification = pickle.load(file)

    def save(self, path):
        if path[-1] != "/":
            path += "/"
        if not os.path.exists(path):
            os.makedirs(path)
        self._save_specification(path)
        self._save_variables_to_directory(path)

    def _restore_from_directory(self, path):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(path + "model.meta")
            saver.restore(self.session, tf.train.latest_checkpoint(path))

    def _make_collections(self):
        # inputs
        self.graph.add_to_collection("x", self.x)
        self.graph.add_to_collection("y", self.y)
        self.graph.add_to_collection("batch_size", self.batch_size)
        self.graph.add_to_collection("seq_len", self.seq_len)
        # outputs
        self.graph.add_to_collection("probabilities", self.probabilities)
        self.graph.add_to_collection("loss", self.loss)
        self.graph.add_to_collection("optimizer", self.optimizer)
        # insertion part
        self.graph.add_to_collection("insertion_probabilities", self.insertion_probabilities)
        if hasattr(self.specification, "sigmoid") and self.specification.sigmoid:
            if hasattr(self, "y_insertion"):
                self.graph.add_to_collection("y_insertion", self.y_insertion)
                self.graph.add_to_collection("insertion_loss", self.insertion_loss)
                self.graph.add_to_collection("insertion_optimizer", self.insertion_optimizer)

    def _remake_graph(self):
        # inputs
        self.x = self.graph.get_collection("x")[0]
        self.y = self.graph.get_collection("y")[0]
        self.batch_size = self.graph.get_collection("batch_size")[0]
        self.seq_len = self.graph.get_collection("seq_len")[0]
        # outputs
        self.probabilities = self.graph.get_collection("probabilities")[0]
        self.loss = self.graph.get_collection("loss")[0]
        self.optimizer = self.graph.get_collection("optimizer")[0]
        # insertion part
        self.insertion_probabilities = self.graph.get_collection("insertion_probabilities")[0]
        if hasattr(self.specification, "sigmoid") and self.specification.sigmoid:
            if len(self.graph.get_collection("y_insertion")) > 0:
                self.y_insertion = self.graph.get_collection("y_insertion")[0]
                self.insertion_loss = self.graph.get_collection("insertion_loss")[0]
                self.insertion_optimizer = self.graph.get_collection("insertion_optimizer")[0]

    def load(self, path):
        if path[-1] != "/":
            path += "/"
        self._load_specification(path)
        self._restore_from_directory(path)
        self._remake_graph()
        self._make_collections()

    def _run_on_sequence(self, operation, sequence):
        return self.session.run(operation,
                                feed_dict={self.x: sequence.reshape((1, sequence.shape[0], sequence.shape[1])),
                                           self.batch_size: 1,
                                           self.seq_len: sequence.shape[0]})[0, :, :]

    def predict(self, sequence):
        return self._run_on_sequence(self.probabilities, sequence)

    def predict_insertion(self, sequence):
        return self._run_on_sequence(self.insertion_probabilities, sequence)


if __name__ == "__main__":
    import numpy as np

    model_path = "saved_models/tmp_bidir"

    batch_size = 10
    seq_len = 20
    dims = 30
    lstm_units = [32]
    dense_units = 16
    negative_examples = True

    data_generator = np.random.RandomState()
    data_generator.seed(42)

    x = data_generator.uniform(0, 1, (batch_size, seq_len, dims))
    y = np.argmax(x, axis=2)
    negative_examples_y = np.zeros((batch_size, seq_len - 1, dims))

    spec = BidirectionalModelSpecification(lstm_units, dense_units, dims, sigmoid=True)
    model = BidirectionalModel()
    #model.initialize(spec)
    model.load(model_path)

    sess = model.session

    epochs = 1000
    for e in range(epochs):
        # train on positive examples
        probs, loss, _ = sess.run([model.probabilities, model.loss, model.optimizer],
                                  feed_dict={model.x: x,
                                             model.y: y,
                                             model.batch_size: batch_size,
                                             model.seq_len: seq_len})
        loss_negex = None
        #print(e, loss)

        # train on negative examples
        if negative_examples:
            loss_negex, _ = sess.run([model.insertion_loss, model.insertion_optimizer],
                                     feed_dict={model.x: x,
                                                model.y_insertion: negative_examples_y,
                                                model.batch_size: batch_size,
                                                model.seq_len: seq_len})
        #print(e, loss_negex)

        if (e + 1) % 100 == 0:
            print()
            print(e, loss, loss_negex)
            correct = 0
            wrong = 0
            for i in range(batch_size):
                for rem_ix in range(1, seq_len - 2):
                    x_rem = np.zeros((1, seq_len - 1, dims))
                    x_rem[0, :rem_ix, :] = x[i, :rem_ix, :]
                    x_rem[0, rem_ix:, :] = x[i, (rem_ix+1):, :]

                    probs = sess.run(model.insertion_probabilities,
                                     feed_dict={model.x: x_rem,
                                                model.batch_size: 1,
                                                model.seq_len: seq_len - 1})
                    pred = np.argmax(probs, axis=2)
                    if pred[0, rem_ix-1] == y[i, rem_ix]:
                        correct += 1
                    else:
                        wrong += 1

            print("%f percent acc (%i / %i)" % (correct / (correct + wrong) * 100, correct, correct + wrong))
            print()

    model.save(model_path)
    model.close_session()
