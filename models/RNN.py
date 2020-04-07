import tensorflow.compat.v1 as tf
from models.BaseNN import *


class RNN(BaseNN):

    def network(self):
        # 4 layers
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [32, 64, 128, 256]]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell, inputs=self.X_tf, dtype=tf.float32)
        

        Z = tf.layers.dense(outputs, self.data_loader.config["n_inputs"], activation="sigmoid")
        
        return Z

    def metrics(self):
        cost = tf.reduce_mean(tf.square(self.y_preds_tf - self.y_tf))
        return cost
