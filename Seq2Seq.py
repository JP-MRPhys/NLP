import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import *

from translation_data import load_vec


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


target_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.de.vec'
source_embedding_path = '/home/jehill/python/NLP/datasets/GloVE/MUSE/wiki.multi.en.vec'

# using existing vocab and embedding
src_embeddings, src_id2word, src_word2id, source_vocab = load_vec(target_embedding_path, nmax=2000000)
tgt_embeddings, tgt_id2word, tgt_word2id, target_vocab = load_vec(source_embedding_path, nmax=2000000)


class Seq2Seq:

    def __init__(self):

        self.ATTENTION_UNITS = 10  # should be equal to decorder units??
        self.encoder_embedding_dim = 300  # add for spacy
        self.decoder_embedding_dim = 300  # add from spacy
        self.batch_size = 256
        self.decoder_vocab_size = 100000
        self.encoder_vocab_size = 100000
        self.lstm_sizes = [128]  # number hidden layer in each LSTM
        self.keep_prob = 0.5

        self.encoder_length = 10  # these are lenght of sentences figure out different for different version
        self.decoder_length = 10  # these are lenght of sentences figure out different for different version

        self.beam_search = False
        self.beam_width = 10

        self.target_start_token = 10  # '<GO>'/start need to feed int32 to the network
        self.target_end_token = 10  # '<END>'/end need to feed int32 to the network

        with tf.variable_scope('rnn_i/o'):

            # use None for batch size and dynamic sequence length
            # embedding to be trained separately, or employ existing embedding (for e.g. from spacy)
            # input's below are assumed gone through embedding layer

            self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.encoder_length, self.batch_size])
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[self.decoder_length, self.batch_size])

            self.source_sequence_length = tf.placeholder(tf.int32, shape=[
                self.batch_size])  # i.e. self.decoder_length use to limit training iter while unrolling .. nneds to be corrected

            self.target_sentence_length = tf.placeholder(tf.int32, shape=[self.batch_size])

        with tf.variable_scope('embeddings'):
            # if not using pre-exisiting embedding ..
            # self.encoder_embeddings=tf.get_variable("encoder_embeddings", [self.decoder_vocab_size,self.encoder_embedding_dim])
            # self.decoder_embeddings=tf.get_variable("decoder_embeddings", [self.encoder_vocab_size,self.decoder_embedding_dim])

            self.encoder_embeddings = tf.get_variable(name="encoder_embedding", shape=np.shape(src_embeddings),
                                                      initializer=tf.constant_initializer(src_embeddings),
                                                      trainable=False)
            self.decoder_embeddings = tf.get_variable(name="decoder_embedding", shape=np.shape(tgt_embeddings),
                                                      initializer=tf.constant_initializer(tgt_embeddings),
                                                      trainable=False)

            self.encoder_input_embeddings = tf.nn.embedding_lookup(self.encoder_embeddings,
                                                                   self.encoder_inputs)  # ouput shape [self.encoder_length, self. batch_siz, emd_dimension (300)]
            self.decoder_input_embeddings = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_inputs)

        """
        with tf.variable_scope('encoder'):
            self.encoder_initial_state, self.encoder_outputs, self.encoder_final_state, self.encoder_cell = build_lstm_layers(self.lstm_sizes,
                                                                                                        self.encoder_input_embeddings,
                                                                                                        self.keep_prob,
                                                                                                        self.batch_size)

        with tf.variable_scope('decoder'):


             self.decoder_initial_state, self.decoder_outputs, self.decoder_final_state, self.decoder_cell = build_lstm_layers(
                    self.lstm_sizes,
                    self.decoder_input_embeddings,
                    self.keep_prob,
                    self.batch_size)
                    

        """
        self.decoder_cell = tf.contrib.rnn.LSTMCell(128)
        self.encoder_cell = tf.contrib.rnn.LSTMCell(128)

        # Getting an initial state of all zeros
        self.encoder_initial_state = self.encoder_cell.zero_state(self.batch_size, tf.float32)

        # perform dynamic unrolling of the network,
        self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(self.encoder_cell,
                                                                           self.encoder_input_embeddings,
                                                                           time_major=True,
                                                                           initial_state=self.encoder_initial_state,
                                                                           dtype=tf.float32)

        self.projection = tf.layers.Dense(self.decoder_vocab_size, use_bias=False)

        with tf.variable_scope('attention'):

            # attention_states: [batch_size, max_time, num_units] #may need to use tf.transpose(self.encoder_outputs, [1, 0, 2]) to feed it to attention (based on time_major)
            self.attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            # self.attention_states=self.encoder_outputs  # when time major is false

            self.attention_mechanism = BahdanauAttention(128,
                                                         self.attention_states)  # pass source_sequence_lenght for improved efficieny ??
            self.attention_cell = AttentionWrapper(self.decoder_cell, self.attention_mechanism, attention_layer_size=64)
            self.attention_decoder_initial_state = self.attention_cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=self.encoder_final_state)

            self.train_helper = TrainingHelper(inputs=self.decoder_input_embeddings,
                                               sequence_length=self.target_sentence_length,
                                               time_major=True)  # need to calculate source sequence length what is this what does
            self.train_decoder = BasicDecoder(self.attention_cell, self.train_helper,
                                              initial_state=self.attention_decoder_initial_state,
                                              output_layer=self.projection)  # using attention
            self.decoded_output_train, self.final_state_attn_decoder, _final_sequence_length = dynamic_decode(
                self.train_decoder)  # see other options here..

            # self.logits = tf.contrib.layers.fully_connected(self.decoded_output_train.rnn_output, self.decoder_vocb_size, activation_fn=None) this was necessary if no projection layer was employed
            self.logits = self.decoded_output_train.rnn_output
            print(self.attention_decoder_initial_state)

            if self.beam_search:

                # replicate initial state beam widht times
                self.beam_search_init_state = tf.contrib.seq2seq.tile_batch(self.attention_decoder_initial_state,
                                                                            multiplier=self.beam_width)

                self.inference_decoder = BeamSearchDecoder(cell=self.decoder_cell, embedding=self.decoder_embeddings,
                                                           start_tokens=tf.fill([self.batch_size],
                                                                                self.target_start_token),
                                                           end_token=self.target_end_token,
                                                           initial_state=self.beam_search_init_state,
                                                           beam_width=self.beam_width,
                                                           output_layer=self.projection, length_penalty_weight=0.0)


            else:
                self.inference_helper = GreedyEmbeddingHelper(embedding=self.encoder_outputs,
                                                              start_tokens=tf.fill([self.batch_size],
                                                                                   self.target_start_token),
                                                              end_token=self.target_end_token)  # need to change decorder embeddeding

                self.inference_decoder = BasicDecoder(self.decoder_cell, self.inference_helper,
                                                      initial_state=self.attention_decoder_initial_state)  # using attention

            self.decoded_output_inference, _, _ = dynamic_decode(self.inference_decoder, output_time_major=True,
                                                                 impute_finished=False,
                                                                 maximum_iterations=100)  # tf.round(self.target_sentence_length*2))

            self.translations = self.decoded_output_inference.predicted_ids

        with tf.variable_scope('rnn_loss'):
            # use cross_entropy as class loss
            self.loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.decoder_inputs,
                logits=self.logits)  # may need to sort out this see the comment in doc.
            self.optimizer = tf.train.AdamOptimizer(0.02)  # .minimize(self.loss) if no gradient clipping required
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.loss, self.params)
            self.clipped_gradients = tf.clip_by_global_norm(self.gradients, clip_norm=5.0)  # how select this value
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.params),
                                                                  global_step=self.global_step)

        with tf.variable_scope('rnn_accuracy'):

            self.accuracy = tf.contrib.metrics.accuracy(
                labels=tf.argmax(self.translated_input, axis=1),
                predictions=self.prediction)

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
        self.saver.save(self.sess, 'model/rnn/seq2seq_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/rnn/seq2seq_%d.ckpt' % (e))


if __name__ == '__main__':
    model = Seq2Seq()
