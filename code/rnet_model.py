from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn

logging.basicConfig(level=logging.INFO)

class RNet(object):

	def __init__(self, FLAGS, id2word, word2id, emb_matrix):
		print 'Initializing RNet Model'
		self.FLAGS = FLAGS
		self.id2word = id2word
		self.word2id = word2id

		#TODO: ADD CHARACTER to id, etc

        # Add all parts of the graph
        with tf.variable_scope("RNet", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        print("ADDING PLACHOLERS")

        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])

        # Character ID's

        # TODO: add char max len

        self.context_ids_c = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.char_max_len])
        self.qn_ids_c = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.char_max_len])

        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix, character_embeddings):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.

          character_embeddings: shape (91, 300)
          	Pretrained character-embeddings vectors, from https://github.com/minimaxir/char-embeddings/blob/master/output/char-embeddings.txt
        """
        print "ADDING EMBEDDINGS"
        with vs.variable_scope("word_embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            e_embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="e_emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.e_context_embs = embedding_ops.embedding_lookup(e_embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.e_qn_embs = embedding_ops.embedding_lookup(e_embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


        """
        with vs.variable_scope("char_embeddings_rnn"):
        	c_embedding_matrix = tf.constant(character_embeddings, dtype=tf.float32, name="c_emb_matrix")
        	self.c_context_embs = embedding_ops.embedding_lookup(c_embedding_matrix, self.context_ids) # shape (batch_size, context_len, max_char_len, embedding_size)
            self.c_qn_embs = embedding_ops.embedding_lookup(c_embedding_matrix, self.qn_ids) # shape (batch_size, question_len, max_char_len, embedding_size)

            context_list = tf.split(self.c_context_embs, self.FLAGS.context_len, axis=1)

            #### TO DO!!!! ADD FLAGS.e_emb_dim
            char_emb_fwd_cell = tf.contrib.rnn.GRUCell(self.FLAGS.e_emb_dim)
            char_emb_back_cell = tf.contrib.rnn.GRUCell(self.FLAGS.e_emb_dim)

            for t in range(self.FLAGS.context_len):
                # Do a BiRNN for Char to get a word_char encoding
                unstacked_context_t = tf.unstack(context_list[t], self.FLAGS.max_char_len, axis= 1) # split ONE WORD into a list of characters
                # REUSE THE VARIABLES
                if t > 0:
                    tf.get_variable_scope().reuse_variables()

                # Not sure if I should use static or dynamic
                output, output_fwd, output_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(char_emb_fwd_cell, char_emb_back_cell, 
                    unstacked_context_t, dtype='float32')

                context_fwd_max = tf.reduce_max(tf.stack(output_fwd, 0), 0) # get forward embedding by max pooling
        """



    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        print "Question and Passage Encoding"

        unstack_context = tf.unstack(self.e_context_embs, self.FLAGS.context_len)
        unstack_qn = tf.unstack(self.e_qn_embs, self.FLAGS.question_len)
        with tf.variable_scope('encoding') as scope:

            # WE CAN CHANGE THE GRU LATER WITH DROPOUT OUR 
            # ADD ENCODE SIZE
            enc_fwd_cell = tf.contrib.rnn.GRUCell(self.FLAGS.encode_size)
            enc_back_cell = tf.contrib.rnn.GRUCell(self.FLAGS.encode_size)
            
            c_output, c_fwd_output, c_back_output = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(emb_fwd_cell, emb_back_cell, 
                    unstacked_context, dtype='float32')

            tf.get_variable_scope().reuse_variables()

            qn_output, q_fwd_output, q_back_output = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(emb_fwd_cell, emb_back_cell, 
                    unstacked_context, dtype='float32')

            u_Q = tf.stack(qn_output, 1) # [batch, q_len, 2 * encode_size] because bidirectional stacks the forward and backward
            u_P = tf.stack(c_output, 1) # [batch, c_len, 2 * encode_size]

        # ADD DROPOUT TO u_Q and u_P

        print "Question-Passage Matching"


        v_P = [] # All attention states across time
        # each element of v_P is an attention state for one time point with dim [batch_size, encode_size]
        for t in range(0, self.FLAGS.context_len): 

            # TODO: MOVE THE VARIABLES TO SOMEWHERE ELSE APPROPRIATE

            W_uQ = tf.get_variable('W_uQ', shape=(2 * self.FLAGS.encode_size, self.FLAGS.encode_size), initializer=tf.contrib.layers.xavier_initializer())
            W_uP = tf.get_variable('W_uP', shape=(2 * self.FLAGS.encode_size, self.FLAGS.encode_size), initializer=tf.contrib.layers.xavier_initializer())
            W_vP = tf.get_variable('W_vP', shape=(self.FLAGS.encode_size, self.FLAGS.encode_size, initializer=tf.contrib.layers.xavier_initializer()))
            v_QP = tf.get_variable('v_QP', shape=(self.FLAGS.encode_size))
            W_g_QP = tf.get_variable('W_g_QP', shape=(4 * self.FLAGS.encode_size, 4 * self.FLAGS.encode_size))

            # TO DO: add drop prob in FLAGS
            QP_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.FLAGS.encode_size), self.FLAGS.drop_prob)
            QP_cell_hidden = QP_cell.zero_state(self.FLAGS.batch_size, dtype=tf.float32)

            WuQ_uQ = tf.tensordot(u_Q, W_uQ, axes = [[2], [0]]) # [batch, q_len, encode_size]
            u_P_t = tf.reshape(u_P[:,t,:], (self.FLAGS.batch_size, 1, -1)) # slice only 1 context word, [batch_size, 1, 2 * encode_size]
            WuP_uP = tf.tensordot(u_P_t, W_uP, axes=[[2],[0]]) # [batch, 1, encode_size]
            
            if t==0:
                s_t = tf.tensordot(tf.tanh(WuQ_uQ + WuP_uP), v_QP, axes=[[2],[0]]) # returns [batch, q_len]
            else:
                v_P_t = tf.reshape(v_P[t-1], (self.FLAGS.batch_size, 1, -1)) # [batch_size, 1, encode_size]
                WvP_vP = tf.tensordot(v_P_t, W_vP, axes=[[2],[0]]) # [batch_size, 1, encode_size]
                s_t = tf.tensordot(tf.tanh(WuQ_uQ + WuP_uP + WvP_vP), v_QP, axes=[[2],[0]]) # returns [batch, q_len]

            a_t = tf.nn.softmax(s_t, 1) # [batch, q_len]
            # [batch, q_len] , [batch,q_len,2*encode_size] -> [batch, 2*encode_size]
            c_t = tf.einsum('ij,ijk->ik', a_t, u_Q) #[batch,2*encode_size]
            
            uPt_ct = tf.concat([tf.squeeze(u_P_t), c_t], 1) # [batch, 2 * 2 * encode_size]
            g_t = tf.nn.sigmoid(tf.matmul(uPt_ct, W_g_QP)) # [batch, 2 * 2 * encode_size]
            uPt_ct_star = tf.einsum('ij,ij->ij', g_t, uPt_ct)

            with tf.variable_scope("QP-Matching") as scope:
                if t > 0:
                    tf.get_variable.reuse_variables()
                    QP_output, QP_cell_hidden = QP_cell(uPt_ct_star, QP_cell_hidden) # both output and hidden [batch_size, encode_size]
                    v_P.append(QP_output)

        v_P = tf.stack(v_P, 1) # [batch, context_len, encode_size]
        v_P = tf.nn.dropout(v_P, self.FLAGS.drop_prob)

        print "SELF MATCHING ATTENTION"

        SM_input = []
        for t in range(0, self.FLAGS.context_len):
            W_v_P = tf.get_variable('W_v_P', shape=(self.FLAGS.encode_size, self.FLAGS.encode_size), initializer=tf.contrib.layers.xavier_initializer())
            W_v_P_tot = tf.get_variable('W_v_P_tot', shape=(self.FLAGS.encode_size, self.FLAGS.encode_size), initializer=tf.contrib.layers.xavier_initializer())

            v_SM = tf.get_variable('v_SM', shape=(self.FLAGS.encode_size))

            v_j_P = tf.reshape(v_P[:,t,:], (self.FLAGS.batch_size, 1, -1)) #Slice 1 v_P in time t [batch_size, 1, encode_size]
            WvP_vj = tf.tensordot(v_j_P, W_v_P, axes=[[2],[0]]) # [batch, 1, encode_size]
            WvPtot_vP = tf.tensordot(v_P, W_v_P_tot, axes=[[2], [0]]) # [batch, context_len, encode_size]

            s_t = tf.tensordot(tf.tanh(WvP_vj + WvPtot_vP), v_SM, axes=[[2],[0]]) # [batch, context_len]
            a_t = tf.nn.softmax(s_t, 1)
            c_t = tf.einsum('ij,ijk->ik', a_t, v_P) #[batch, encode_size]

            # add the gate
            vPt_ct = tf.concat([tf.squeeze(v_j_P), c_t], 1) #[batch, 2 * encode_size]
            g_t = tf.nn.sigmoid(vPt_ct)
            vPt_ct_star = tf.einsum('ij,ij->ij', g_t, vPt_ct) # [batch, 2*encode_size]

            SM_input.append(vPt_ct_star)

        # Someone here just stacked and then unstack, not sure why so I will just directly use SM_input

        with tf.variable_scope("self_matching") as scope:
            SM_fwd_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.FLAGS.encode_size), self.FLAGS.drop_prob)
            SM_back_cel = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(self.FLAGS.encode_size), self.FLAGS.drop_prob)
            SM_outputs, SM_final_fwd, SM_final_back = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(SM_fwd_cell, SM_back_cell, SM_input, dtype=tf.float32)
            h_P = tf.stack(SM_outputs, 1)

        h_P = tf.nn.dropout(h_P, self.FLAGS.drop_prob)

        print "OUTPUT LAYER"

    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size

            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
