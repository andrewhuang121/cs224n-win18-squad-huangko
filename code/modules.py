# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell

class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist

class BiDAF_attn(object):
    def __init__(self, keep_prob, query_vec_size, context_vec_size):
        self.keep_prob = keep_prob
        self.query_vec_size = query_vec_size
        self.context_vec_size = context_vec_size

    def build_graph(self, context, context_mask, query, query_mask):
        '''
        context: should be [batch, context_length, context_encode_size]
        querys: should be  [batch, query_length, query_encode_size]
        context_mask: [batch_size, context_length]
        querys_mask: [batch_size, query_length]
        '''
        with tf.variable_scope("BIDAF_attn"):
            encode_size = context.shape[2]

            w1T = tf.get_variable('w1T', shape=(encode_size), initializer=tf.contrib.layers.xavier_initializer())
            w2T = tf.get_variable('w2T', shape=(encode_size), initializer=tf.contrib.layers.xavier_initializer())
            w3T = tf.get_variable('w3T', shape=(encode_size), initializer=tf.contrib.layers.xavier_initializer())

            query_t = tf.reshape(tf.matmul(context, w1T), (-1, 1, self.query_vec_size))
            context_t = tf.reshape(tf.matmul(querys, w2T), (-1, self.context_vec_size, 1))
            context_dot_query_t = tf.einsum('iaj,ibj->iab', tf.multiply(context, w3T), query)
            similarity = query_t + context_t + context_dot_query_t # should be [batch, context_length, query_length]

            with tf.variable_scope("Context2Query"):
                #softmax over query
                query_logits_mask = tf.expand_dims(query_mask, 1) #[batch, 1, query_length] ::: adding the same mask at each context
                _, attn_query = masked_softmax(similarity, query_logits_mask, 2) # [batch, context_len, query_len]
                U_tild = tf.einsum('ijk,ikl->ijl', attn_query, query) #[batch, context_len, encode] each context word selects a combination of query words

            with tf.variable_scope("Query2Context"):
                _, attn_context = masked_softmax(tf.reduce_max(similarity, 2), context_mask, 1) # [batch, context_len], [batch, context_len] -> [batch, context_len]
                H_tild = tf.einsum('ij,ijl->il', attn_context, context) #[batch, encode]
                H_tild = tf.tile(tf.expand_dims(H_tild, 1), [1, context.shape[1],1]) # [batch, context_len, encode]

            # combine attentions
            G = tf.concat([context, U_tild, tf.multiply(context, U_tild), tf.multiply(context, H_tild)], 2) #[batch, context_len, 4 * encode]

            return G

class ModelingLayer(object):

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.g0_fwd = DropoutWrapper(rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_prob)
        self.g0_back = DropoutWrapper(rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_prob)
        self.g1_fwd = DropoutWrapper(rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_prob)
        self.g1_back = DropoutWrapper(rnn_cell.LSTMCell(self.hidden_size), input_keep_prob=self.keep_prob)
        self.multi_fwd = MultiRNNCell([self.g0_fwd, self.g1_fwd])
        self.multi_back = MultiRNNCell([self.g0_back, self.g1_back])

    def build_graph(self, inputs, masks):
        #inputs should be [batch, context_len, some encode_size]
        # mask [batch, context_len]

        input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

        # Note: fw_out and bw_out are the hidden states for every timestep.
        # Each is shape (batch_size, seq_len, hidden_size).
        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.multi_fwd, self.multi_back, inputs, input_lens, dtype=tf.float32)

        # Concatenate the forward and backward hidden states
        M = tf.concat([fw_out, bw_out], 2)

        # Apply dropout
        M = tf.nn.dropout(out, self.keep_prob)

        return M

class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "querys" and "context" (see lectures).
    In the terminology of "X attends to Y", "querys attend to context".

    In the baseline model, the querys are the context hidden states
    and the context are the question hidden states.

    We choose to use general terminology of querys and context in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, query_vec_size, context_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          query_vec_size: size of the query vectors. int
          context_vec_size: size of the context vectors. int
        """
        self.keep_prob = keep_prob
        self.query_vec_size = query_vec_size
        self.context_vec_size = context_vec_size

    def build_graph(self, context, context_mask, querys):
        """
        querys attend to context.
        For each query, return an attention distribution and an attention output vector.

        Inputs:
          context: Tensor shape (batch_size, num_context, context_vec_size).
          context_mask: Tensor shape (batch_size, num_context).
            1s where there's real input, 0s where there's padding
          querys: Tensor shape (batch_size, num_querys, context_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_querys, num_context).
            For each query, the distribution should sum to 1,
            and should be 0 in the context locations that correspond to padding.
          output: Tensor shape (batch_size, num_querys, hidden_size).
            This is the attention output; the weighted sum of the context
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            context_t = tf.transpose(context, perm=[0, 2, 1]) # (batch_size, context_vec_size, num_context)
            attn_logits = tf.matmul(querys, context_t) # shape (batch_size, num_querys, num_context)
            attn_logits_mask = tf.expand_dims(context_mask, 1) # shape (batch_size, 1, num_context)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_querys, num_context). take softmax over context

            # Use attention distribution to take weighted sum of context
            output = tf.matmul(attn_dist, context) # shape (batch_size, num_querys, context_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
