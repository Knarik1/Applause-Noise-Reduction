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

    def create_placeholders(self):
        # tf.disable_eager_execution()
        self.X_tf = tf.placeholder(tf.float32, shape=[None, self.data_loader.config["seq_length"], self.data_loader.config["n_inputs"]], name="noisy")
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.data_loader.config["seq_length"], self.data_loader.config["n_inputs"]], name='signal_mask')
        self.training_flag = tf.placeholder_with_default(False, shape=[], name='training_flag')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def initialize_network(self):
        # opening Session
        self.sess = tf.Session()
        
        # initializing Saver for checkpoints
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
        
        # initializing FileWriter
        self.summary_op = tf.summary.merge_all()   # merges all summaries : val + train
        if self.summaries_path != "":
            self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.summaries_path, "train"), self.sess.graph)
            self.val_summary_writer = tf.summary.FileWriter(os.path.join(self.summaries_path,"val"), self.sess.graph)

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
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"]).minimize(self.cost, global_step=self.global_step)

        self.optimizer = optimizer

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        # number of iterations
        num_iter_train = np.ceil(self.data_loader.m_train / self.data_loader.config["train_batch_size"])
        num_iter_val = np.ceil(self.data_loader.m_val / self.data_loader.config["val_batch_size"])
        
        for epoch in range(self.config["num_epochs"]):
            print("----Start training----")
            minibatch_cost_sum_train = 0

            # for every epoch shuffle entire train dataset
            np.random.shuffle(self.data_loader.train_paths.T)

            for iter in tqdm(range(int(num_iter_train))):
                X_batch, y_batch = self.data_loader.train_data_loader(iter)

                feed_dict_train = {
                    self.X_tf: X_batch,
                    self.y_tf: y_batch,
                    self.training_flag: True
                }

                _, minibatch_cost, global_step, train_summary = self.sess.run([self.optimizer, self.cost, self.global_step, self.summary_op], feed_dict=feed_dict_train)
                
                print("global_step")
                print(global_step)
                print("minibatch_cost")
                print(minibatch_cost)

                minibatch_cost_sum_train += minibatch_cost
                
                # mean cost for each iter
                minibatch_cost_mean_train = minibatch_cost_sum_train / num_iter_train
                
                if iter % summary_step == 0:
                    # train summary
                    self.train_summary_writer.add_summary(train_summary, global_step=global_step)
                        
                
            print("----Start validation----")
            minibatch_cost_sum_val = 0
    
            # for every epoch shuffle entire val dataset
            np.random.shuffle(self.data_loader.val_paths.T)
    
            for iter in tqdm(range(int(num_iter_val))):
                X_val_batch, y_val_batch = self.data_loader.val_data_loader(iter)
    
                feed_dict_val = {
                    self.X_tf: X_val_batch,
                    self.y_tf: y_val_batch,
                    self.training_flag: False
                }
    
                minibatch_cost_val, val_summary = self.sess.run([self.cost, self.summary_op], feed_dict=feed_dict_val)
    
                print("minibatch_cost val")
                print(minibatch_cost_val)

                minibatch_cost_sum_val += minibatch_cost_val
    
                if iter % summary_step == 0:
                    # val summary
                    self.val_summary_writer.add_summary(val_summary, global_step=global_step)
    
                # mean cost for each epoch
                minibatch_cost_mean_val = minibatch_cost_sum_val / num_iter_val

            if epoch % display_step == 0:
                print('Epoch %d: Train Cost = %.4f' % (epoch, minibatch_cost_mean_train))

            if epoch % validation_step == 0:
                print('Epoch %d: Val Cost = %.4f' % (epoch, minibatch_cost_mean_val))
                
            if epoch % checkpoint_step == 0:
                self.saver.save(self.sess, os.path.join(self.checkpoints_path, 'epoch' + ".ckpt"), global_step=global_step)

    def test_model(self):
        minibatch_cost_sum = 0
        num_iter = np.ceil(self.data_loader.m_test / self.data_loader.config["test_batch_size"])
        
        for iter in tqdm(range(int(num_iter))):
            X_test_batch, y_test_batch = self.data_loader.test_data_loader(iter)

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
        # config
        preprocess.SUBSET = "test",
        preprocess.NOISE_DATAPATH_TEST = preprocess.DATA_PATH + '/Applause_test'

        # preprocess wav as ffts
        magn_batch, phase_batch = preprocess.process_test_song(input_path)
        magn_ideal_batch, phase_ideal_batch = preprocess.process_test_song('./data/Songs/test_song/Carlos Gonzalez - A Place For Us/mixture16.wav')


        # feed_dict_test = {
        #     self.X_tf: magn_batch,
        #     self.training_flag: False
        # }
        
        # pred_mask_batch = self.sess.run(self.y_preds_tf, feed_dict=feed_dict_test)
        # pred_mask_batch = np.clip(magn_ideal_batch / (magn_batch + 10e-7), 0, 1)
        pred_mask_batch = magn_ideal_batch**2 /(magn_ideal_batch + magn_batch)**2
        
        # estimate magnitudes
        magn_estimates_batch = pred_mask_batch * magn_batch
        
        # build full song
        estim_song = np.zeros(preprocess.SLICE_STEP * (magn_estimates_batch.shape[0] + 1))
        
        for sl in range(magn_estimates_batch.shape[0]):
            # get slices
            estim_song_slice = preprocess.get_signal_from_fft(magn_estimates_batch[sl], phase_batch[sl])
            estim_song[sl * preprocess.SLICE_STEP: sl * preprocess.SLICE_STEP + preprocess.SLICE_DURATION * preprocess.SAMPLE_RATE] += estim_song_slice * (preprocess.vorbis_window(len(estim_song_slice)) ** 2)
        
        # handle clipping
        max_absolute_estim_song = np.max(np.abs(estim_song))
        if max_absolute_estim_song > 32767:
            estim_song = estim_song * (32767 / max_absolute_estim_song)

        estim_name = os.path.basename(input_path)[:-4] + "_estimated.wav" 
        wavfile.write(output_path + '/' + estim_name, preprocess.SAMPLE_RATE, estim_song.astype("int16"))
        print("----Test song has generated successfully!----") 

    @abstractmethod
    def network(self):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self):
        raise NotImplementedError('subclasses must override metrics()!')
