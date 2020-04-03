import tensorflow as tf
import os
import pickle
from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell


class UniDirectionalModelSpecification:
    def __init__(self, lstm_units, dense_units, dims, name):
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dims = dims
        self.name = name

    def __str__(self):
        return str({'lstm_units': self.lstm_units,
                    'dense_units': self.dense_units,
                    'dims': self.dims,
                    'name': self.name})


class UniDirectionalModel:
    def __init__(self):
        self.specification = None

    def _make_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # input
            self.batch_size = tf.placeholder(tf.int32, [])
            self.seq_len = tf.placeholder(tf.int32)
            self.x = tf.placeholder(tf.float32, shape=(None, None, self.specification.dims))  # batch_size, seq_len, dim
            self.y = tf.placeholder(tf.int32, shape=(None, None))  # batch_size, seq_len
            self.keep_prob = tf.placeholder(tf.float32, shape=())

            # LSTM cell
            self.lstm_cell = MultiRNNCell(
                [BasicLSTMCell(units) for units in self.specification.lstm_units]
            )
            self.initial_hidden_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

            # recurrent part
            self.rnn_output, self.hidden_state = tf.nn.dynamic_rnn(
                cell=self.lstm_cell,
                inputs=self.x,
                initial_state=self.initial_hidden_state
            )

            # dropout
            self.rnn_dropout = tf.nn.dropout(self.rnn_output, self.keep_prob)

            # dense part
            afunc = tf.nn.relu
            self.dense = tf.layers.dense(self.rnn_dropout,
                                         self.specification.dense_units[0],
                                         activation=afunc)
            self.dense_dropout = tf.nn.dropout(self.dense, self.keep_prob)
            for i in range(1, len(self.specification.dense_units)):
                self.dense = tf.layers.dense(self.dense_dropout,
                                             self.specification.dense_units[i],
                                             activation=afunc)
                self.dense_dropout = tf.nn.dropout(self.dense, self.keep_prob)

            # softmax output
            self.activation = tf.layers.dense(self.dense_dropout, self.specification.dims)
            self.probabilities = tf.nn.softmax(self.activation)

            # loss
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y,
                                                               logits=self.activation)

            # optimizer
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # open session
        self.session = tf.Session(graph=self.graph)

    def _make_collections(self):
        # inputs
        self.graph.add_to_collection("x", self.x)
        self.graph.add_to_collection("y", self.y)
        self.graph.add_to_collection("seq_len", self.seq_len)
        self.graph.add_to_collection("batch_size", self.batch_size)
        self.graph.add_to_collection("keep_prob", self.keep_prob)
        # hidden state
        for i in range(len(self.specification.lstm_units)):
            self.graph.add_to_collection("initial_hidden_state.h", self.initial_hidden_state[i].h)
            self.graph.add_to_collection("initial_hidden_state.c", self.initial_hidden_state[i].c)
            self.graph.add_to_collection("hidden_state.h", self.hidden_state[i].h)
            self.graph.add_to_collection("hidden_state.c", self.hidden_state[i].c)
        # outputs
        self.graph.add_to_collection("probabilities", self.probabilities)
        self.graph.add_to_collection("loss", self.loss)
        self.graph.add_to_collection("optimizer", self.optimizer)

    def _initialize_variables(self):
        with self.graph.as_default():
            init = tf.global_variables_initializer()
        self.session.run(init)

    def initialize(self, specification):
        self.specification = specification
        self._make_graph()
        self._make_collections()
        self._initialize_variables()

    def close_session(self):
        self.session.close()

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

    def _remake_graph(self):
        # inputs:
        self.x = self.graph.get_collection("x")[0]
        self.y = self.graph.get_collection("y")[0]
        self.seq_len = self.graph.get_collection("seq_len")[0]
        self.batch_size = self.graph.get_collection("batch_size")[0]
        if len(self.graph.get_collection("keep_prob")) > 0:
            self.keep_prob = self.graph.get_collection("keep_prob")[0]
        else:
            with self.graph.as_default():
                self.keep_prob = tf.placeholder(tf.float32)
        # hidden state:
        initial_hidden_state_tuples = []
        hidden_state_tuples = []
        for i in range(len(self.specification.lstm_units)):
            initial_hidden_state_tuples.append(
                tf.contrib.rnn.LSTMStateTuple(self.graph.get_collection("initial_hidden_state.c")[i],
                                              self.graph.get_collection("initial_hidden_state.h")[i])
            )
            hidden_state_tuples.append(
                tf.contrib.rnn.LSTMStateTuple(self.graph.get_collection("hidden_state.c")[i],
                                              self.graph.get_collection("hidden_state.h")[i])
            )
        self.initial_hidden_state = tuple(initial_hidden_state_tuples)
        self.hidden_state = tuple(hidden_state_tuples)
        # outputs:
        self.probabilities = self.graph.get_collection("probabilities")[0]
        self.loss = self.graph.get_collection("loss")[0]
        self.optimizer = self.graph.get_collection("optimizer")[0]

    def load(self, path):
        if path[-1] != "/":
            path += "/"
        self._load_specification(path)
        self._restore_from_directory(path)
        self._remake_graph()
        self._make_collections()

    def predict_batch(self, x):
        return self.session.run(self.probabilities,
                                feed_dict={self.x: x,
                                           self.seq_len: x.shape[1],
                                           self.batch_size: x.shape[0],
                                           self.keep_prob: 1.0})

    def predict(self, x):
        X = x.reshape(1, x.shape[0], x.shape[1])
        return self.predict_batch(X)[0, :, :]


if __name__ == "__main__":
    import numpy as np

    # specification
    epochs = 100
    batch_size = 20
    seq_len = 16
    dims = 10
    lstm_units = [32, 16]
    dense_units = [32]
    spec = UniDirectionalModelSpecification(lstm_units, dense_units, dims, "bla")

    # random data
    X = np.random.uniform(0, 1, (batch_size, seq_len, dims))
    y = np.random.choice(range(dims), size=(batch_size, seq_len), replace=True)

    # init model
    model = UniDirectionalModel()
    model.initialize(spec)
    sess = model.session

    # train
    for e in range(epochs):
        loss, _ = sess.run([model.loss, model.optimizer],
                           feed_dict={model.batch_size: batch_size,
                                      model.x: X,
                                      model.y: y,
                                      model.keep_prob: 1})

        print(e, loss)

    # predict
    bs_valid = 1
    sl_valid = 19
    X_valid = np.random.uniform(0, 1, (bs_valid, sl_valid, dims))
    probs = sess.run(model.probabilities,
                     feed_dict={model.batch_size: bs_valid,
                                model.x: X_valid,
                                model.keep_prob: 1})
    print(probs)
    print(np.sum(probs, axis=2))
    print(probs.shape)

    model.save("saved_models/tmp")

    sess.close()
