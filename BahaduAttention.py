"""
#Create an attention mechanism
We're using Bahdanau attention. Lets decide on notation before writing the simplified form:

FC = Fully connected (dense) layer
EO = Encoder output
H = hidden state
X = input to the decoder
And the pseudo-code:

score = FC(tanh(FC(EO) + FC(H)))
attention weights = softmax(score, axis = 1). Softmax by default is applied on the last axis but here we want to apply it on the 1st axis, since the shape of score is (batch_size, max_length, 1). Max_length is the length of our input. Since we are trying to assign a weight to each input, softmax should be applied on that axis.
context vector = sum(attention weights * EO, axis = 1). Same reason as above for choosing axis as 1.
embedding output = The input to the decoder X is passed through an embedding layer.

merged vector = concat(embedding output, context vector)
This merged vector is then given to the decorder..
"""

# hidden_state=needs to be the and then needs to be called in the dynamic un-rolling of the network, understand this bit better...

score = tf.contrib.tf.nn.tanh(
    fully_connected_layer(self.encoder_outputs, self.ATTENSION_UNTS) * self.encoder_outputs + fully_connected_layer(
        self.hidden_state, self.ATTENTION_UNTS))
self.score = fully_connected_layer(self.score, self.ATTENTION_UNITS)
self.attention_weights = tf.nn.softmax(self.score, axis=1)
self.context_vector = self.attention_weights * self.encoder_outputs
self.context_vector = tf.reduce_sum(self.context_vector, axis=1)
self.d_embedding = embedding_layer(self.context_vector)

# add stuff on time major of swaps see the attention.py, it also W1 and W2 and V, which are learned during training, this can be easily wrapper around a wrapper using the new tensorflow API

"""
