from tensorflow.contrib.rnn import LSTMCell


class multiLayerLSTM:

    def __int__(self):
        self.number_units = 10

        cell = LSTMCell(self.number_units)
