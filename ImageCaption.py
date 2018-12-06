import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from VGG16 import vgg16_cnn_emb  # this is the VGG encoder
from translation_data import load_glove

glove_embeddings, glove_id2word, glove_word2id, glove_vocab = load_glove()




class ImageCaption:

    def __init__(self):

        self.vgg_weights_dir = './model/VGG/'
        self.word_embedding_dim = 300
        self.keep_prob = 0.5
        self.batch_size = 512
        self.lstm_sizes = [128]  # number hidden layer in each LSTM
        self.num_classes = 2
        self.ATTENSION_UNIT = 10
        self.decoder_vocab_size = 200000  # change to the vocab size i.e

        # create the inputs
        self.input_image = tf.placeholder(tf.float32, [None, None, None,
                                                       None])  # variable input in [BATCH_SIZE, width, height, depth]
        self.input_image_244 = tf.image.resize_images(self.input_image, [244, 244])  # resize images for VGG

        self.input_caption = tf.placeholder(tf.int32, [None, None])
        self.input_mask = None
        self.target_caption = None

        # VGG (or your favourite CNN) encoder
        self.vgg_net, self.conv4 = vgg16_cnn_emb(self.input_image_244)
        print(self.vgg_net)
        print(self.conv4)

        self.encoder_outputs = tf.placeholder(tf.float32, shape=[None,
                                                                 8 * 8 * 512])  # tf.reshape(self.conv4, [tf.shape(self.conv4)[0], -1])  #may need to reshape based on the dimensions

        # embedding for the input sentense
        self.decoder_embeddings = tf.get_variable(name="decoder_embedding", shape=np.shape(glove_embeddings),
                                                  initializer=tf.constant_initializer(glove_embeddings),
                                                  trainable=False)

        # self.sequence_length=tf.reduce_sum(self.input_mask,1) #input mask to keep the sequence length
        self.decoder_input_embeddings = tf.nn.embedding_lookup(self.decoder_embeddings,
                                                               self.input_caption)  # these are grounds truths for the words

        self.lstm_zero_state, self.lstm_cell = self.build_lstm_layers(self.lstm_sizes, self.keep_prob, self.batch_size)

        # run the lstm to obtain initial
        self.initial_state = self.lstm_cell(self.encoder_outputs, self.lstm_zero_state)  # these are image embeddings

        # perform dynamic unrolling of the network
        self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(self.lstm_cell, self.decoder_input_embeddings,
                                                                initial_state=self.initial_state)
        self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, self.lstm_cell.output_size])
        self.logits = fully_connected(self.lstm_outputs, num_outputs=self.decoder_vocab_size)

        self.target_labels = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                             self.decoder_length])  # need to figure this out difference between target captions and input cations ....

        # may need to sort out this see the comment in doc
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.target_labels,
            logits=self.logits)

        self.optimizer = tf.train.AdamOptimizer(0.02)  # .minimize(self.loss) if no gradient clipping required

        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss, self.params)
        self.clipped_gradients = tf.clip_by_global_norm(self.gradients, clip_norm=5.0)  # how select this value
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.train_operation = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.params),
                                                              global_step=self.global_step)

        """
        attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
        source_sequence_length = 100;  # this needs to come from input length ..
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.ATTENSION_UNIT, attention_states,memory_sequence_length=source_sequence_length)

        # we then need to pass the information via the AttensionWrapper to combine
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism, attention_layer_size=self.ATTENSION_UNIT)
        """

    def load_VGG_weights(self, sess):
        # load weights for VGG
        # load weights

        npz = np.load(self.vgg_weights_dir + 'vgg16_weights.npz')
        vgg_weights = []
        for idx, val in enumerate(sorted(npz.items())[0:20]):
            print("  Loading pre trained VGG16, CNN part %s" % str(val[1].shape))
            vgg_weights.append(self.vgg_net.all_params[idx].assign(val[1]))

        print("Completed loading VGG weights")
        sess.run(vgg_weights)
        print("Completed applying VGG weights to current session")

    def build_lstm_layers(self, lstm_sizes, keep_prob, batch_size):
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
        drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob) for lstm in lstms]

        # Stack up multiple LSTM layers
        cell = tf.contrib.rnn.MultiRNNCell(drops)

        # Getting an initial state of all zeros
        zero_state = cell.zero_state(batch_size, tf.float32)

        return zero_state, cell


if __name__ == '__main__':

    tf.reset_default_graph()
    model = ImageCaption()

    # with tf.Session() as sess:
    # model.load_VGG_weights(sess)
