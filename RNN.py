import tensorflow as tf
import numpy as np

import spacy
import keras as keras
from keras.datasets import imdb as imdb

from tensorflow.contrib.rnn import LSTMCell, RNNCell, BasicLSTMCell, BasicRNNCell



if __name__ == '__main__':

    nlp =spacy.load('en')


    lstm_cell=LSTMCell(num_units=128)



    output,state=tf.nn.dynamic_rnn(lstm_cell,tf.constant(np.float32(np.random.rand(20,2000,3))),dtype=tf.float32)



    a=nlp('test')
    print(np.shape(a.vector))
    print(output)
    print(state)





