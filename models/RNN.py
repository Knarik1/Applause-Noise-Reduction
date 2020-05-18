import tensorflow.compat.v1 as tf
from models.BaseNN import *


class RNN(BaseNN):

    def _rnn_placeholders(self, state, c_name):
        c = state
        c = tf.placeholder_with_default(c, c.shape, c_name)
        # h = tf.placeholder_with_default(h, h.shape, h_name)
        # return tf.contrib.rnn.LSTMStateTuple(c, h)
        return c   

    def network(self):
        self.num_units = 1500
        batch_x = self.X_tf
        # self.n_layers = 2
        seq_length = [self.data_loader.config["seq_length"]]*self.data_loader.config["train_batch_size"]
        with tf.variable_scope("network"):
            gru_layer_fw = tf.contrib.rnn.GRUCell(self.num_units)
            gru_layer_bw = tf.contrib.rnn.GRUCell(self.num_units)
                #gru_layer = tf.contrib.rnn.GRUCell(self.num_units)
                # multi_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=1.0) for _ in range(self.n_layers)]) 
            initial_state_fw = self._rnn_placeholders(gru_layer_fw.zero_state(tf.shape(batch_x)[0], tf.float32), "c_state_fw")
            initial_state_bw = self._rnn_placeholders(gru_layer_bw.zero_state(tf.shape(batch_x)[0], tf.float32), "c_state_bw")
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(gru_layer_fw, gru_layer_bw, batch_x, sequence_length=seq_length, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype="float32")
            outputs = tf.concat(outputs, 2)
            dense1 = tf.layers.dense(inputs=outputs, units=3000, activation=tf.nn.relu, name="dense1")
            lin1 = tf.layers.dense(inputs=outputs, units=3000, activation=None, name="lin1")
            sum1 = dense1 + lin1
            dense2 = tf.layers.dense(inputs=sum1, units=3000, activation=tf.nn.relu, name="dense2")
            lin2 = tf.layers.dense(inputs=sum1, units=3000, activation=None, name="lin2")
            sum2 = dense2 + lin2
            prediction = tf.layers.dense(inputs=sum2, units=self.data_loader.config["n_inputs"], activation=tf.sigmoid, name="last_dense")
        return prediction

    # def network(self):
    #     outputs, state = tf.nn.dynamic_rnn(cell=tf.nn.rnn_cell.LSTMCell(1000), inputs=self.X_tf, dtype=tf.float32)
    #     Z = tf.layers.dense(outputs, self.data_loader.config["n_inputs"], activation='sigmoid')
        
    #     return Z

    def metrics(self):
        cost = tf.reduce_mean(tf.square(self.y_preds_tf - self.y_tf))
        tf.summary.scalar('cost_funtion', cost)
        return cost 

       
