import tensorflow.compat.v1 as tf
from models.RNN import  *
from preprocess import prepare_dataset, generate_noisy_signal

tf.app.flags.DEFINE_string('mode', 'train', 'whether to train/test/preprocess')

tf.app.flags.DEFINE_string('data_path', 'data', 'path to your preprocessed CSV data file')
tf.app.flags.DEFINE_string('subset', 'train', 'path to your CSV file linking paths of mixes and sources')
tf.app.flags.DEFINE_float('split_valid', 0, 'validation set split')
tf.app.flags.DEFINE_string('save_as', 'h5', 'Wheter to save as h5/wav')
tf.app.flags.DEFINE_integer('slice_duration', '5', 'Duration in seconds of slice to be cut before stft')
tf.app.flags.DEFINE_integer('workers', '2', 'Number of workers')

tf.app.flags.DEFINE_integer('train_batch_size', 512, 'number of elements in a training batch')
tf.app.flags.DEFINE_integer('val_batch_size', 512, 'number of elements in a validation batch')
tf.app.flags.DEFINE_integer('test_batch_size', 512, 'number of elements in a testing batch')

tf.app.flags.DEFINE_integer('n_inputs', 241, 'Fourier transform coefficients -> 1+n_fft/2')
tf.app.flags.DEFINE_integer('seq_length', 332, 'Number of frames')

tf.app.flags.DEFINE_integer('num_epochs', 1000, 'epochs to train')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer')

tf.app.flags.DEFINE_integer('display_step', 1, 'Number of steps we cycle through before displaying detailed progress.')
tf.app.flags.DEFINE_integer('validation_step', 1, 'Number of steps we cycle through before validating the model.')

tf.app.flags.DEFINE_string('base_dir', './results', 'Directory in which results will be stored.')
tf.app.flags.DEFINE_integer('checkpoint_step', 1, 'Number of steps we cycle through before saving checkpoint.')
tf.app.flags.DEFINE_integer('max_to_keep', 50, 'Number of checkpoint files to keep.')

tf.app.flags.DEFINE_integer('summary_step', 1, 'Number of steps we cycle through before saving summary.')

tf.app.flags.DEFINE_string('signal_path', None, 'signal path to generate noisy signal')
tf.app.flags.DEFINE_boolean('return_noise', False, 'save also generated noisy signal noise')
tf.app.flags.DEFINE_string('noisy_song_path', None, 'path to load noisy song')
tf.app.flags.DEFINE_string('output_estimated_path', './results/RNN', 'path to save estimated song')
tf.app.flags.DEFINE_string('model_name', 'RNN', 'name of model')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    if FLAGS.mode == 'preprocess':
        prepare_dataset(FLAGS.data_path, FLAGS.subset, FLAGS.split_valid, FLAGS.slice_duration, FLAGS.save_as)
        exit()

    if FLAGS.mode == 'generate_noisy':
        generate_noisy_signal(FLAGS.signal_path, FLAGS.return_noise)
        exit()    

    model = RNN(
        train_dir=FLAGS.data_path + '/train',
        val_dir=FLAGS.data_path + '/valid',
        test_dir=FLAGS.data_path + '/test',
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        test_batch_size=FLAGS.test_batch_size,
        n_inputs=FLAGS.n_inputs,
        seq_length=FLAGS.seq_length,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.learning_rate,
        base_dir=FLAGS.base_dir,
        max_to_keep=FLAGS.max_to_keep,
        model_name=FLAGS.model_name
    )

    model.create_network()
    model.initialize_network()

    if FLAGS.mode == 'train':
        model.train_model(FLAGS.display_step, FLAGS.validation_step, FLAGS.checkpoint_step, FLAGS.summary_step)
    elif FLAGS.mode == 'test':
        model.test_model()
    elif FLAGS.mode == 'test_song':
        model.estimate_test_song(FLAGS.noisy_song_path, FLAGS.output_estimated_path)


if __name__ == "__main__":
    tf.app.run()
