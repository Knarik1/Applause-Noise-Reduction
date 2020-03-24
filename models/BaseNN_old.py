import tensorflow.compat.v1 as tf
from data_loader import *
import numpy as np
from abc import abstractmethod
import pandas as pd
import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import WaveDataset

class BaseNN:
    def __init__(self, train_signals_dir, test_signals_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, n_inputs, seq_length,
                 learning_rate, base_dir, max_to_keep, model_name, model):

    
        self.config = {
            "train_signals_dir": train_signals_dir,
            "test_signals_dir": test_signals_dir,
            "n_inputs": n_inputs,
            "seq_length": seq_length,
            "train_batch_size": train_batch_size,
            "test_batch_size": test_batch_size,
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

    def create_network(self):
        # clears the default graph stack and resets the global default graph.
        tf.reset_default_graph()

        self.create_placeholders()
        self.pred_mask_tf = self.network()
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
        tf.disable_eager_execution()
        self.mix_tf = tf.placeholder(tf.float32, shape=[None, self.config["seq_length"], self.config["n_inputs"]], name='mix')
        self.signal_mask_tf = tf.placeholder(tf.float32, shape=[None, self.config["seq_length"], self.config["n_inputs"]], name='signal_mask')
        self.pred_mask_tf = tf.placeholder(tf.float32, shape=[None, self.config["seq_length"], self.config["n_inputs"]], name='pred_signal_mask')
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
        loss = self.metrics()
        self.cost = tf.reduce_mean(loss)

    def create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"]).minimize(self.cost)

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
        train_data = pd.read_csv(self.config["train_signals_dir"])

        transforms_to_do = [transforms.Normalize(), transforms.ToTensor()]

        dataset = WaveDataset(train_data,
                              # transforms=[transforms.HorizontalCrop(128),
                              # transforms.Normalize()],
                              # use_log_scale = use_log_scale)
                              transforms=transforms_to_do,
                              use_log_scale=True)
        dataloader = DataLoader(dataset, batch_size=self.config["train_batch_size"],
                                shuffle=True, num_workers=2)

        for e in range(self.config["num_epochs"]):
            try:
                print('Starting Epoch', str(e) + '/' + str(self.config["num_epochs"]))
                epoch_full_loss = 0
                for n_track, lst in enumerate(tqdm(dataloader)):
                    # TODO change source hardcoding, handle unequal size of mix and source
                    normalized_mix = lst[1].float().numpy()
                    normalized_signal = lst[0].float().numpy()
                    original_mix = lst[2].float().numpy()
                    signal = lst[3].float().numpy()

                    # Put 0.0 where values are low
                    #
                    b = np.divide(normalized_signal, normalized_mix, out=np.zeros_like(normalized_signal), where=normalized_mix!=0.0)
                    
                    z = b.reshape(len(b),-1).flatten()
                    v = np.where(z == np.max(z))
                    idx = v[0][0]
                    print("signal value")
                    print(normalized_signal.reshape(len(normalized_signal),-1).flatten()[idx])
                    print(np.min(normalized_signal))
                    print(np.max(normalized_signal))
                    print(normalized_signal[:20])
                    # print(signal.reshape(len(signal),-1).flatten()[idx])


                    print("mix value")
                    print(normalized_mix.reshape(len(normalized_mix),-1).flatten()[idx])
                    print(np.min(normalized_mix))
                    print(np.max(normalized_mix))
                    print(normalized_mix[:20])
                    # print(original_mix.reshape(len(original_mix),-1).flatten()[idx])
                    print(np.max(z))


                    exit()

                    signal_mask = np.divide(normalized_signal, normalized_mix, out=np.zeros_like(normalized_signal), where=normalized_mix!=0.0)


                    feed_dict_train = {
                        self.mix_tf: np.swapaxes(normalized_mix, 2, 1),
                        self.signal_mask_tf: np.swapaxes(signal_mask, 2, 1),
                        self.training_flag: True
                    }

                    _, minibatch_cost, train_summary = self.sess.run([self.optimizer, self.cost, self.train_summary_merged], feed_dict=feed_dict_train)
                    epoch_full_loss += minibatch_cost
                    print("minibatch-----------------------", minibatch_cost)

                    # if iter % summary_step == 0:
                    # train summary
                    self.train_writer.add_summary(train_summary, e * self.config["train_batch_size"] + n_track)


                epoch_mean_loss = epoch_full_loss / len(dataloader)
                print('Epoch completed, Training Loss: ', epoch_mean_loss)

                if e % checkpoint_step == 0:
                    self.saver.save(self.sess, self.checkpoints_path + "/checkpoint_" + str(e)+".ckpt")

                self.train_writer.close()
                # self.val_writer.close()
            except KeyboardInterrupt:
                    pass


           
    def test_model(self):
        test_data = pd.read_csv(self.config["test_signals_dir"])

        transforms_to_do = [transforms.Normalize(), transforms.ToTensor()]

        dataset = WaveDataset(test_data,
                              # transforms=[transforms.HorizontalCrop(128),
                              # transforms.Normalize()],
                              # use_log_scale = use_log_scale)
                              transforms=transforms_to_do,
                              use_log_scale=True)
        dataloader = DataLoader(dataset, batch_size=self.config["test_batch_size"],
                                shuffle=True, num_workers=2)

        try:
            full_loss = 0
            for n_track, lst in enumerate(tqdm(dataloader)):
                # TODO change source hardcoding, handle unequal size of mix and source
                normalized_mix = lst[1].float().numpy()
                normalized_signal = lst[0].float().numpy()
                original_mix = lst[2].float().numpy()
                signal = lst[3].float().numpy()
                #
                # b = np.divide(normalized_signal, normalized_mix, out=np.zeros_like(normalized_signal), where=normalized_mix!=0.0)
                # h = signal/(original_mix)
                #
                # z = b.reshape(len(b),-1).flatten()
                # v = np.where(z == np.max(z))
                # idx = v[0][0]
                # print("signal value")
                # print(normalized_signal.reshape(len(normalized_signal),-1).flatten()[idx])
                # print(np.min(normalized_signal))
                # print(np.max(normalized_signal))
                # print(normalized_signal[:20])
                # print(normalized_signal.reshape(len(normalized_signal),-1).flatten()[idx])
                # print(signal.reshape(len(signal),-1).flatten()[idx])


                # print("mix value")
                # print(normalized_mix.reshape(len(normalized_mix),-1).flatten()[idx])
                # print(np.min(normalized_mix))
                # print(np.max(normalized_mix))
                # print(normalized_mix[:20])
                # # print(original_mix.reshape(len(original_mix),-1).flatten()[idx])
                # print(np.max(z))


                # exit()

                signal_mask = np.divide(normalized_signal, normalized_mix, out=np.zeros_like(normalized_signal), where=normalized_mix!=0.0)


                feed_dict_test = {
                    self.mix_tf: np.swapaxes(normalized_mix, 2, 1),
                    self.signal_mask_tf: np.swapaxes(signal_mask, 2, 1),
                    self.training_flag: False
                }

                minibatch_cost = self.sess.run([self.cost, output], feed_dict=feed_dict_test)
                full_loss += minibatch_cost
                print("minibatch-----------------------", minibatch_cost)

                output = self.pred_mask_tf * self.mix_tf


            epoch_mean_loss = full_loss / len(dataloader)
            print('Epoch completed, Testing Loss: ', epoch_mean_loss)


        except KeyboardInterrupt:
                pass


    @abstractmethod
    def network(self):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self):
        raise NotImplementedError('subclasses must override metrics()!')