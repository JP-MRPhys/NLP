import tensorflow as tf


class SentimentReviewRNN:

    def __init__(self):
        with tf.variable_scope('rnn_i/o'):
            # use None for batch size and dynamic sequence length
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, 384])
            self.groundtruths = tf.placeholder(tf.float32, shape=[None, 2])

        with tf.variable_scope('rnn_cell'):
            self.cell = tf.contrib.rnn.LSTMCell(128)
            # self.stackcell=rnn_cell.MultiRNNCell([self.cell]*3, state_is_tuple=True)
            self.out_cell = tf.contrib.rnn.OutputProjectionWrapper(self.cell, 2)
            # project RNN output into target class dimension

        with tf.variable_scope('rnn_forward'):
            # use dynamic_rnn for different length
            self.outputs, _ = tf.nn.dynamic_rnn(
                self.out_cell, self.inputs, dtype=tf.float32)
            self.outputs2 = self.outputs[:, -1, :]  # only use the last output of sequence

        with tf.variable_scope('rnn_loss'):
            # use cross_entropy as class loss
            self.loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.groundtruths, logits=self.outputs2)
            self.optimizer = tf.train.AdamOptimizer(0.02).minimize(self.loss)

        with tf.variable_scope('rnn_accuracy'):
            self.accuracy = tf.contrib.metrics.accuracy(
                labels=tf.argmax(self.groundtruths, axis=1),
                predictions=tf.argmax(self.outputs2, axis=1))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())  # don't forget to initial all variables
        self.saver = tf.train.Saver()  # a saver is for saving or restoring your trained weight

    def train(self, batch_x, batch_y):
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        # feed in input and groundtruth to get loss and update the weight via Adam optimizer
        loss, accuracy, _ = self.sess.run(
            [self.loss, self.accuracy, self.optimizer], fd)

        return loss, accuracy

    def test(self, batch_x, batch_y):
        fd = {}
        fd[self.inputs] = batch_x
        fd[self.groundtruths] = batch_y
        prediction, accuracy = self.sess.run([self.outputs2, self.accuracy], fd)

        return prediction, accuracy

    def save(self, e):
        self.saver.save(self.sess, 'model/rnn/rnn_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, 'model/rnn/rnn_%d.ckpt' % (e))


if __name__ == '__main__':
    # hyperparameter of our network
    EPOCHS = 20

    tf.reset_default_graph()
    model = SentimentReviewRNN()

    """

    DataDir = '/home/jehill/python/NLP/datasets/'

    train_dir = os.path.join(DataDir, 'train')
    test_dir = os.path.join(DataDir, 'test')

    train_data = get_imbd_data(train_dir)
    test_data = get_imbd_data(test_dir)

    n_train=len(train_data)
    rec_loss = []

    for epoch in range(EPOCHS):

      train_data = train_data.sample(frac=1).reset_index(drop=True)
      loss_train = 0
      accuracy_train = 0

      for index, row in train_data.iterrows():

        train_X = sentence_embedding(row['text'])
        train_Y = row['sentiment']

        # no batch until now but need to do that soon reshape in the batch format in the mean time and feed the network
        if train_Y: sentiment=np.array([0,1])
        else: sentiment=np.array([1,0])

        sentiment=np.reshape(sentiment, [1,2])
        print(np.shape(sentiment))

        print(np.shape(train_X))
        train_X=np.reshape(train_X, [1,len(train_X), 384])

        print(np.shape(train_X))


        loss_batch, accuracy_batch = model.train(train_X, sentiment)
        loss_train += loss_batch
        accuracy_train += accuracy_batch
        # b=b+BATCH_SIZE

      loss_train /= n_train
      accuracy_train /= n_train

      model.save(epoch)  # save your model after each epoch
      rec_loss.append([loss_train, accuracy_train])
      print("Training completed")

  # np.save('./model/rnn/rec_loss.npy', rec_loss)


  
    rec_loss = []

    for e in range(EPOCHS):  # train for several epochs
      loss_train = 0
      accuracy_train = 0

      for b in range(len(train_X)):  # feed batches one by one
        batch_x = train_X[b]
        batch_y = (train_Y[b])
        batch_x=np.reshape(batch_x, [1, len(batch_x),1])


        print(np.shape(batch_x))
        print(batch_y)

        if batch_y: sentiment=np.array([0,1])
        else: sentiment=np.array([1,0])

        sentiment=np.reshape(sentiment, [1,2])
        loss_batch, accuracy_batch = model.train(batch_x, sentiment)

        loss_train += loss_batch
        accuracy_train += accuracy_batch
        #b=b+BATCH_SIZE

      loss_train /= n_train
      accuracy_train /= n_train

      model.save(e)  # save your model after each epoch
      rec_loss.append([loss_train, accuracy_train])
      print("Training completed")

    #np.save('./model/rnn/rec_loss.npy', rec_loss)


    """
