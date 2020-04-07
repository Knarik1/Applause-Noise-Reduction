import tensorflow as tf
from data_loader import *
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from abc import abstractmethod
import preprocess as preprocess
import time

class BaseNN:
    def __init__(self, train_dir, val_dir, test_dir, train_batch_size,
                 val_batch_size, test_batch_size,
                 n_inputs, seq_length,
                 num_epochs, learning_rate,
                 base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_dir, val_dir, test_dir,
                                      train_batch_size, val_batch_size, test_batch_size,
                                      n_inputs, seq_length)
                             
        self.config = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "base_dir": base_dir,
            "max_to_keep": max_to_keep,
            "model_name": model_name
        }

        save_path = os.path.join(self.config["base_dir"], self.config["model_name"])
        self.checkpoints_path = os.path.join(save_path, "checkpoints")
        self.summaries_path = os.path.join(save_path, "summaries")

        # check if checkpoints' directory exists, otherwise create it.
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        # check if summaries' directory exists, otherwise create it.
        if not os.path.exists(self.summaries_path):
            os.makedirs(self.summaries_path)
            
        print("learning rate: ", self.config["learning_rate"])    
        print("train batch size: ", self.data_loader.config["train_batch_size"])     

    def create_network(self):
        # clears the default graph stack and resets the global default graph.
        tf.reset_default_graph()

        self.create_placeholders()
        self.y_preds_tf = self.network()
        self.compute_cost()
        self.create_optimizer()

    def init_saver(self):
        # initialize Saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])

    def init_fwriter(self):
        with tf.name_scope('summaries_train'):
            tf.summary.scalar("train_cost", self.cost)
            self.train_summary_merged = tf.summary.merge_all()

        with tf.name_scope('summaries_val'):
            tf.summary.scalar("val_cost", self.cost)
            self.val_summary_merged = tf.summary.merge_all()

        # initialize FileWriter to save summaries
        self.train_writer = tf.summary.FileWriter(self.summaries_path + '/train', self.sess.graph)
        self.val_writer = tf.summary.FileWriter(self.summaries_path + '/val', self.sess.graph)

    def create_placeholders(self):
        # tf.disable_eager_execution()
        self.X_tf = tf.placeholder(tf.float32, shape=[None, self.data_loader.config["seq_length"], self.data_loader.config["n_inputs"]], name="noisy")
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.data_loader.config["seq_length"], self.data_loader.config["n_inputs"]], name='signal_mask')
        self.training_flag = tf.placeholder_with_default(False, shape=[], name='training_flag')

    def initialize_network(self):
        # opening Session
        self.sess = tf.Session()

        # initializing Saver
        self.init_saver()

        # initializing FileWriter
        self.init_fwriter()

        ckpt = tf.train.get_checkpoint_state(self.checkpoints_path)

        # restore/init weights
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("----Restore params----")
        else:
            self.sess.run(tf.global_variables_initializer())
            print("----Init params----")
        
    def compute_cost(self):
        self.cost = self.metrics()
        
    def create_optimizer(self):
        # update ops for batch normalization
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"]).minimize(self.cost)

        self.optimizer = optimizer

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        # number of iterations
        num_iter_train = np.ceil(self.data_loader.m_train / self.data_loader.config["train_batch_size"])
        num_iter_val = np.ceil(self.data_loader.m_val / self.data_loader.config["val_batch_size"])

        print("----Start training----")
        for epoch in range(self.config["num_epochs"]):
            # ------------------------------------- Train ---------------------------------------
            start_time_train = time.time()
            minibatch_cost_sum_train = 0

            # getting new sequence for every epoch to shuffle entire train dataset
            perm = np.random.permutation(self.data_loader.m_train)
            start_time_train_loader = time.time()
            X_batch, y_batch = self.data_loader.train_data_loader(0, perm=perm)
            print("Loader(train) --- %s seconds ---" % (time.time() - start_time_train_loader))
            for iter in range(int(num_iter_train)):
                X_batch, y_batch = self.data_loader.train_data_loader(iter, perm=perm)

                feed_dict_train = {
                    self.X_tf: X_batch,
                    self.y_tf: y_batch,
                    self.training_flag: True
                }

                _, minibatch_cost, train_summary = self.sess.run([self.optimizer, self.cost, self.train_summary_merged], feed_dict=feed_dict_train)

                minibatch_cost_sum_train += minibatch_cost

                if iter % summary_step == 0:
                    # train summary
                    self.train_writer.add_summary(train_summary, epoch * self.data_loader.config["train_batch_size"] + iter)

            # mean cost for each epoch
            minibatch_cost_mean_train = minibatch_cost_sum_train / num_iter_train
            print("Training --- %s seconds ---" % (time.time() - start_time_train))


            # ------------------------------------- Validation ---------------------------------------
            start_time_val = time.time()
            minibatch_cost_sum_val = 0

            # getting new sequence for every epoch to shuffle entire val dataset
            perm_val = np.random.permutation(self.data_loader.m_val)

            for iter in range(int(num_iter_val)):
                X_val_batch, y_val_batch = self.data_loader.val_data_loader(iter, perm=perm_val)

                feed_dict_val = {
                    self.X_tf: X_val_batch,
                    self.y_tf: y_val_batch,
                    self.training_flag: False
                }

                minibatch_cost, val_summary = self.sess.run([self.cost, self.val_summary_merged], feed_dict=feed_dict_val)

                minibatch_cost_sum_val += minibatch_cost

                if iter % summary_step == 0:
                    # val summary
                    self.val_writer.add_summary(val_summary, epoch * self.data_loader.config["val_batch_size"] + iter)

            # mean cost for each epoch
            minibatch_cost_mean_val = minibatch_cost_sum_val / num_iter_val
            print("Val --- %s seconds ---" % (time.time() - start_time_val))

            if epoch % display_step == 0:
                print('Epoch %d: Train Cost = %.4f' % (epoch, minibatch_cost_mean_train))

            if epoch % validation_step == 0:
                print('Epoch %d: Val Cost = %.4f' % (epoch, minibatch_cost_mean_val))
                
            if epoch % checkpoint_step == 0:
                start_time_ckpt = time.time()
                self.saver.save(self.sess, self.checkpoints_path + "/checkpoint_" + str(epoch)+".ckpt")
                print("Checkpoint saver --- %s seconds ---" % (time.time() - start_time_ckpt))

        self.train_writer.close()
        self.val_writer.close()

    def test_model(self):
        minibatch_cost_sum = 0

        perm = np.random.permutation(self.data_loader.m_test)
        num_iter = np.ceil(self.data_loader.m_test / self.data_loader.config["test_batch_size"])

        for iter in tqdm(range(int(num_iter))):
            X_test_batch, y_test_batch = self.data_loader.test_data_loader(iter, perm=perm)

            feed_dict_test = {
                self.X_tf: X_test_batch,
                self.y_tf: y_test_batch,
                self.training_flag: False
            }

            minibatch_cost = self.sess.run(self.cost, feed_dict=feed_dict_test)

            minibatch_cost_sum += minibatch_cost

        # mean cost
        test_cost = minibatch_cost_sum / num_iter

        print('Test Cost = ', test_cost)

    def estimate_test_song(self, input_path, output_path):
        # predict masks
        magn_batch, phase_batch = self.data_loader.test_song_loader(input_path)

        feed_dict_test = {
            self.X_tf: magn_batch,
            self.training_flag: False
        }
        
        pred_mask_batch = self.sess.run(self.y_preds_tf, feed_dict=feed_dict_test)
        
        # estimate magnitudes
        magn_estimates_batch = pred_mask_batch * magn_batch
        
        # build full song
        estim_song = np.zeros(preprocess.SLICE_DURATION * preprocess.SAMPLE_RATE * magn_estimates_batch.shape[0])

        for sl in range(magn_estimates_batch.shape[0]):
            # get slices
            estim_song_slice = preprocess.get_signal_from_fft(magn_estimates_batch[sl], phase_batch[sl])
            estim_song[sl * preprocess.SAMPLE_RATE * preprocess.SLICE_DURATION: (sl + 1) * preprocess.SLICE_DURATION * preprocess.SAMPLE_RATE] += estim_song_slice

        wavfile.write(output_path + "/estimated_song.wav", preprocess.SAMPLE_RATE, estim_song.astype("int16"))
        print("----Test song has generated successfully!----")

    @abstractmethod
    def network(self):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self):
        raise NotImplementedError('subclasses must override metrics()!')