import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from translation_data import load_glove

glove_embeddings, glove_id2word, glove_word2id, glove_vocab = load_glove()


# time major: where encoder length comes first before the batch size, this will influence model specific e.g. attention see below for more details


def build_lstm_layers(lstm_sizes, embed_input, keep_prob_, batch_size):
    """
    Create the LSTM layers
    inputs: array containing size of hidden layer for each lstm,
            input_embedding, for the shape batch_size, sequence_length, emddeding dimension [None, None, 384], None and None are to handle variable batch size and variable sequence length
            keep_prob for the dropout and batch_size

    outputs: initial state for the RNN (lstm) : tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)] .. only two here e.g. [(256,128), (256,64)]
             outputs of the RNN [Batch_size, sequence_length, last_hidden_layer_dim]
             RNN cell: tensorflow implementation of the RNN cell
             final state: tuple of [(batch_size, hidden_layer_1), (batch_size, hidden_layer_2)]

    """
    lstms = [tf.contrib.rnn.LSTMCell(size) for size in lstm_sizes]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

    # Stack up multiple LSTM layers
    cell = tf.contrib.rnn.MultiRNNCell(drops)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

    # perform dynamic unrolling of the network,
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, time_major=True, initial_state=initial_state)

    return initial_state, lstm_outputs, final_state, cell


def score():
    return


# 2. Construct the decoder cell
def create_cell(rnn_size, keep_prob):
    # lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))

    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    return drop


# we just modify the Seq2Seq model by replacing the encodering embedding with feature vectors from image model and weights...

class Seq2Seq:

    def __init__(self):
        self.ATTENTION_UNITS = 10  # should be equal to decorder units??
        self.encoder_embedding_dim = 300  # add for  spacy
        self.decoder_embedding_dim = 300  # add from spacy
        self.batch_size = 256
        self.decoder_vocab_size = 200000
        self.encoder_vocab_size = 200000
        self.lstm_sizes = [128, 128]  # number hidden layer in each LSTM
        self.keep_prob = 0.5

        self.rnn_size = 128
        self.num_rnn_layers = 2

        self.encoder_length = 10  # these are length of sentences figure out different for different version
        self.decoder_length = 10  # these are length of sentences figure out different for different version

        self.beam_search = True
        self.beam_width = 10

        self.target_start_token = 10  # '<GO>' need to feed int32 to the network
        self.target_end_token = 10  # '<END>' need to feed int32 to the network

        with tf.variable_scope('rnn_i/o'):
            # use None for batch size and dynamic sequence length
            # embedding to be trained separately, or employ existing embedding (for e.g. from spacy)
            # input's below are assumed gone through embedding layer

            self.image_features = tf.placeholder(tf.float32, shape=[self.batch_size, 8 * 8 * 512])
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.decoder_length])  # -tokens

            self.source_sequence_length = tf.placeholder(tf.int32, shape=[
                self.batch_size])  # i.e. self.decoder_length use to limit training iter while unrolling .. nneds to be corrected
            self.target_sentence_length = tf.placeholder(tf.int32, shape=[self.batch_size])

        with tf.variable_scope('embeddings'):
            # these are when once wants to train the embedding from scratch ..
            # self.encoder_embeddings=tf.get_variable("encoder_embeddings", [self.decoder_vocab_size,self.encoder_embedding_dim])
            # self.decoder_embeddings=tf.get_variable("decoder_embeddings", [self.encoder_vocab_size,self.decoder_embedding_dim])

            # embedding for the input sentense
            self.decoder_embeddings = tf.get_variable(name="decoder_embedding", shape=np.shape(glove_embeddings),
                                                      initializer=tf.constant_initializer(glove_embeddings),
                                                      trainable=False)

            self.decoder_input_embeddings = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs)

        with tf.variable_scope('decoder') as decoder:
            self.dec_cell = tf.contrib.rnn.MultiRNNCell(
                [create_cell(rnn_size, self.keep_prob) for rnn_size in self.lstm_sizes])

            # self.dec_cell = tf.contrib.rnn.LSTMCell(128)

            self.dec_zero_state = self.dec_cell.zero_state(self.batch_size, tf.float32)

            _, self.dec_initial_state = self.dec_cell(self.image_features,
                                                      self.dec_zero_state)  # these are image embeddings

            print(self.dec_zero_state)
            print(self.dec_initial_state)

            decoder.reuse_variables()

            # perform dynamic unrolling of the network
            self.lstm_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, self.decoder_input_embeddings,
                                                     initial_state=self.dec_initial_state, dtype=tf.float32)
            self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, self.lstm_cell.output_size])
            self.logits = fully_connected(self.lstm_outputs, num_outputs=self.decoder_vocab_size)

            print("logits")
            print(self.logits)
            self.projection = tf.layers.Dense(self.decoder_vocab_size, use_bias=False)

            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
            self.saver = tf.train.Saver()  # a saver is for saving or restoring your trained weight

    def inference(self, batch_x, batch_y, batch_size):
        """
         NEED TO RE-WRITE this function interface by adding the state
        :param batch_x:
        :param batch_y:
        :return

        """

        # restore the model

        # with tf.Session() as sess:
        #    model=model.restore();

        # test_state = model.cell.zero_state(batch_size, tf.float32)
        """
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        fd[self.initial_state] = test_state
        prediction, accuracy = self.sess.run([self.prediction, self.accuracy], fd)

        return prediction, accuracy
        """

    def train(self):
        return

    def save(self, e):
        self.saver.save(self.sess, 'model/rnn/image2text_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/rnn/image2text_%d.ckpt' % (e))


if __name__ == '__main__':
    model = Seq2Seq()
