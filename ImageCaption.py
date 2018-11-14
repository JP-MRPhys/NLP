import numpy as np
import tensorflow as tf

from VGG16 import vgg16_cnn_emb  # this is the VGG encorder


def build_lstm_layers(self, lstm_sizes, embed_input, keep_prob_, batch_size, scope):
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

    with tf.VariableScope(scope):
        lstms = [tf.contrib.rnn.LSTMCell(size) for size in lstm_sizes]
        # Add dropout to the cell
        drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

        # Stack up multiple LSTM layers
        cell = tf.contrib.rnn.MultiRNNCell(drops)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

        # perform dynamic unrolling of the network,
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_input, initial_state=initial_state)

    return initial_state, lstm_outputs, final_state, cell


class ImageCaption:

    def __init__(self):
        self.vgg_weights_dir = './model/VGG/'
        self.word_embedding_dim = 300
        self.keep_prob = 0.5
        self.batch_size = 512
        self.lstm_sizes = [128, 64]  # number hidden layer in each LSTM
        self.num_classes = 2
        self.ATTENSION_UNIT = 10

        self.input_image = tf.placeholder(tf.float32, [None, None, None,
                                                       None])  # variable input in [BATCH_SIZE, width, height, depth]
        self.input_image_244 = tf.image.resize_images(self.input_image, [244, 244])  # resize images for VGG

        self.vgg_net, self.conv4 = vgg16_cnn_emb(self.input_image_244)
        print(self.vgg_net)
        print(self.conv4)

        self.encoder_ouputs = self.vgg_net

        self.decoder_initial_state, self.decoder_outputs, self.decoder_final_state, self.decoder_cell = build_lstm_layers(
            self.lstm_sizes,
            self.self.encoder_outputs,  # or is this an or the final state
            self.keep_prob,
            self.batch_size, scope='decorder')

        attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
        source_sequence_length = 100;  # this needs to come from input lenght..
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.ATTENSION_UNIT, attention_states,
                                                                   memory_sequence_length=source_sequence_length)

        # we then need to pass the information via the AttensionWrapper to combine
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism,
                                                           attention_layer_size=self.ATTENSION_UNIT)

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

    def demo_eric(self):
        print("Does nothing..")
        return


if __name__ == '__main__':
    tf.reset_default_graph()
    model = ImageCaption()

    # with tf.Session() as sess:
    # model.load_VGG_weights(sess)
