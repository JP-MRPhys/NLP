from ms_coco_dataset import *

glove_embeddings = np.random.rand(20000, 300)


# glove_embeddings, glove_id2word, glove_word2id, glove_vocab = load_glove()


class image2text():

    def __init__(self, pre_trained_embedding):
        self.batch_size = 64
        self.max_seq_length = 30
        self.image_dim1 = 244  # these are input dim for VGG
        self.image_dim2 = 244  # these are input dim for VGG
        self.image_embedding_dim1 = 196  # these are the VGG output dimensions
        self.image_embedding_dim2 = 512  # these are the VGG output dimensions
        self.lstm_dim = 128
        self.keep_prob = 0.5  # dropout rate
        self.lstm_hidden_units = 128  # size of the attention model
        self.attention_units = 128
        self.start_token = 'start'  # start to token to start predicting the words
        self.word_embeddings = pre_trained_embedding
        self.decoder_vocab_size = np.shape(pre_trained_embedding)[
            0]  # these needs to be vocab size of the dropout we are not applying softmax on the output while training
        self.word_embeddings_dim = np.shape(pre_trained_embedding)[
            1]  # these needs to be vocab size of the dropout we are not applying softmax on the output while training
        self.EPOCHS = 20
        self.BATCH_SIZE = 50

        # hyper parameter of our network can be passed via config file, can be passed via config file and word emedding be loaded directly

        self.training = tf.placeholder(tf.bool, shape=())

        self.images = tf.placeholder(tf.float32, shape=[None, self, ])

        self.caption_input = tf.placeholder(tf.int32, shape=[None, self.max_seq_length])  # this is none and none

        # we shift the captions to get target words used for calc the loss along with decoder logits
        self.target_words = self.caption_input[:, 1:]

        self.image_features = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_embedding_dim1,
                                                                self.image_embedding_dim2])

        # self.mask=tf.to_float(tf.not_equal(self.target_words),self._null)  #unknown token
        # embedding for the input sentence
        self.decoder_embeddings = tf.get_variable(name="decoder_embedding", shape=np.shape(self.word_embeddings),
                                                  initializer=tf.constant_initializer(self.word_embeddings),
                                                  trainable=False)  # we are just using existing embedding and not training them

        self.decoder_cell = self.create_cell(self.lstm_hidden_units, self.keep_prob)

        self.image_features_mean = tf.reduce_mean(self.image_features, 1)  # [B,IMD_D2]
        initial_memory, initial_hidden_state = self.initialize_lstm_state(self.image_features_mean)
        initial_state = initial_memory, initial_hidden_state

        # apply attention to get attention context
        if self.training:

            # training mode

            self.decoder_input_embeddings = tf.nn.embedding_lookup(self.decoder_embeddings, self.caption_input)
            self.loss = 0.0
            # generate word one by one

            for time_index in range(self.max_seq_length - 1):
                # apply attention
                attention_context, _ = self.BahdanauAttention(self.image_features, initial_hidden_state)
                lstm_input = tf.concat([self.decoder_input_embeddings[:, time_index, :], attention_context], 1)
                lstm_output, (c, h) = self.decoder_cell(lstm_input, initial_state)
                initial_state = (c, h)
                initial_hidden_state = h

                self.loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_words[:, time_index],
                                                                            logits=lstm_output)

            self.optimizer = tf.train.AdamOptimizer(0.02).minimize(self.loss)  # no gradient clipping !!

            """
            #need to fix the error for mergeing shapes for applying gradient clipping

            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.loss, self.params)
            self.clipped_gradients = tf.clip_by_global_norm(self.gradients, clip_norm=5.0)  # how select this value
            self.grads_and_vars =list(zip(self.clipped_gradients,self.params))
            self.train_op=self.optimizer.apply_gradients(grads_and_vars=self.grads_and_vars)
            """

        else:

            # inference mode
            self.predicted_words_list = []

            for time_index in range(self.max_seq_length - 1):

                if time_index == 0:

                    number_samples = tf.shape(self.image_features)[0]
                    start_samples = tf.fill(number_samples, self.start_token)
                    x = tf.nn.embedding_lookup(self.decoder_input_embeddings, start_samples)

                else:

                    print("add predicted word embedding")
                    x = tf.nn.embedding_lookup(self.decoder_input_embeddings, predicted_word)

                attention_context, _ = self.BahdanauAttention(self.image_features, initial_hidden_state)

                lstm_input = tf.concat([x, attention_context], 1)

                lstm_output, (c, h) = self.decoder_cell(lstm_input, initial_state)

                initial_state = (c, h)
                initial_hidden_state = h

                # lstm_output is the logits, we use that to generate the next word

                predicted_word = tf.argmax(lstm_output, 1)
                self.predicted_words_list.append(predicted_word)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()  # a saver is for saving or restoring your trained weight

    # 2. Construct the decoder cell
    def create_cell(self, rnn_size, keep_prob):

        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, state_is_tuple=True)
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        return drop

    def initialize_lstm_state(self, features_mean):
        memory = tf.layers.dense(features_mean, self.lstm_dim)
        output = tf.layers.dense(features_mean, self.lstm_dim)

        return memory, output

    def BahdanauAttention(self, features, hidden_state):

        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)

        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_state_with_time_axis = tf.expand_dims(hidden_state, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(
            tf.layers.dense(features, self.attention_units) + tf.layers.dense(hidden_state_with_time_axis,
                                                                              self.attention_units))
        score = tf.layers.dense(score, 1)

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def train_batch(self, batch_images, batch_captions):

        batch_features = self.VGG_image_features(batch_images)

        feed_dict = {}

        feed_dict[self.image_features] = batch_features
        feed_dict[self.caption_input] = batch_captions
        feed_dict[self.istraining] = True

        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict)

        return loss

    def inference_batch(self, batch_images):

        batch_features = self.VGG_image_features(batch_images)

        feed_dict = {}

        feed_dict[self.image_features] = batch_features
        feed_dict[self.istraining] = False

        _ = self.sess.run([self.optimizer], feed_dict)

        return self.predicted_words_list

    def save(self, e):
        self.saver.save(self.sess, 'model/image2text_attention_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/image2text_attention_%d.ckpt' % (e))

    def train_coco(self):

        # get coco datasets

        # initialize all the variables before training
        self.sess.run(tf.global_variables_initializer())


if __name__ == '__main__':

    # load the glove embedding
    glove_vocab, glove_embeddings, word_embedding_dict = load_glove()

    tf.reset_default_graph()
    model = image2text(glove_embeddings)

    rec_loss = []

    for epoch in range(EPOCHS):

        # get image filenames and captions for coco dataset in build shuffling for
        captions, image_filenames = get_coco_datasets();

        loss_train = 0
        accuracy_train = 0
        istraining = True

        number_training_points = len(image_filenames)

        for idx in range(0, number_training_points, BATCH_SIZE):
            filenames = image_filename[idx:idx + BATCH_SIZE]
            caption = captions[idx:idx + BATCH_SIZE]
            batch_images, batch_captions = get_batch_data_image_caption(filenames, caption, word_embedding_dict,
                                                                        max_sequence_length=model.max_seq_length)

            loss_batch, _ = model.train(batch_images, batch_captions, istraining)
            loss_train += loss_batch

            print("EPOCH: " + str(epoch) + "BATCH_INDEX:" + str(idx) + "Batch Loss:" + str(loss_batch))

        loss_train /= number_training_points
        rec_loss.append([loss_train, accuracy_train])

        if (epoch % 5 == 0):
            model.save(epoch)  # save your model after every 5 epoch

    np.save('./model/image2text_attention/rec_loss.npy', rec_loss)
    print("Training completed")
