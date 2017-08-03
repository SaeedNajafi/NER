import re
import os
import codecs
import getpass
import sys
import time
import numpy as np
import tensorflow as tf
import utils as ut

class NER(object):
    """Implements an NER (Named Entity Recognition) model.
    """

    """Model hyperparams and data information.
    """
    word_embedding_size = 100
    char_embedding_size = 25
    word_hidden_units = 100
    char_hidden_units = 25

    max_sentence_length = 150
    max_word_length = 25
    tag_size = 17

    batch_size = 16
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 24
    early_stopping = 2
    CRF = True

    #path to different files!
    word_dic_path = './data/glove_en_word_100_dic.txt'
    word_vectors_path = './data/glove_en_word_100_vectors.txt'
    char_dic_path = './data/en_char_dic.txt'
    train_set_path = './data/eng.train.v1'
    dev_set_path = './data/eng.testa.v1'
    test_set_path = './data/eng.testb.v1'

    def load_data(self):
        """
        Loads starter word vectors, creates random character vectors,
        and loads train/dev/test data.

        """

        # Loads the starter word vectors
        print "INFO: Reading word embeddings!"
        self.word_vectors, words = ut.load_embeddings(
                                        self.word_dic_path,
                                        self.word_vectors_path)

        #Adding new words of the training set and their random vectors.
        new_words = []
        temp_dic={}
        with open(self.train_set_path) as fd:
            for line in fd:
                if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                    continue
                else:
                    word = line.strip().split("\t")[0]
                    if (word not in words) and (word.lower() not in words):
                        if re.search(r'\d', word) is None:
                            if word in temp_dic.keys():
                                temp_dic[word] += 1
                            else:
                                temp_dic[word] = 1

        for (new_word, count) in sorted(temp_dic.items(), key=lambda item:item[1]):
            if count>1:
                new_words.append(new_word)

        words = words + new_words
        #Initializing new word vectors!
        boundry = np.sqrt(np.divide(3.0, self.word_embedding_size))
        new_word_vectors = np.random.uniform(
                                low=-boundry,
                                high=boundry,
                                size=(len(new_words), self.word_embedding_size))

        self.word_vectors = np.vstack([self.word_vectors, new_word_vectors])
        self.num_to_word = dict(enumerate(words))
        self.word_to_num = {v:k for k,v in self.num_to_word.iteritems()}

        # Load the starter char vectors
        print "INFO: Reading character embeddings!"
        _, chars = ut.load_embeddings(
                        self.char_dic_path,
                        None)

        self.num_to_char = dict(enumerate(chars))
        self.char_to_num = {v:k for k,v in self.num_to_char.iteritems()}

	# For IOBES format
	tagnames = ['O', 'B-LOC', 'I-LOC', 'S-LOC', 'E-LOC',
                    'B-ORG', 'I-ORG', 'S-ORG', 'E-ORG',
                    'B-PER','I-PER', 'S-PER', 'E-PER',
                    'B-MISC', 'I-MISC', 'S-MISC', 'E-MISC']

        self.num_to_tag = dict(enumerate(tagnames))
        self.tag_to_num = {v:k for k,v in self.num_to_tag.iteritems()}

        #Loads the training set
        print "INFO: Reading training data!"
        docs = ut.load_dataset(self.train_set_path)
        Char_X_train, Cap_X_train, Word_X_train, Y_train = ut.docs_to_sentences(
                                                                docs,
                                                                self.word_to_num,
                                                                self.tag_to_num,
                                                                self.char_to_num)

        out = ut.padding(
                        Char_X_train,
                        Cap_X_train,
                        Word_X_train,
                        Y_train,
                        self.max_sentence_length,
                        self.max_word_length)

        self.char_X_train = out['char_X']
        self.word_length_X_train = out['word_length_X']
        self.cap_X_train = out['cap_X']
        self.word_X_train = out['word_X']
        self.mask_X_train = out['mask_X']
        self.sentence_length_X_train = out['sentence_length_X']
        self.Y_train = out['Y']

        # Loads the dev set (for tuning hyperparameters)
        print "INFO: Reading dev data!"
        docs = ut.load_dataset(self.dev_set_path)
        Char_X_dev, Cap_X_dev, Word_X_dev, Y_dev = ut.docs_to_sentences(
                                                        docs,
                                                        self.word_to_num,
                                                        self.tag_to_num,
                                                        self.char_to_num)

        out = ut.padding(
                        Char_X_dev,
                        Cap_X_dev,
                        Word_X_dev,
                        Y_dev,
                        self.max_sentence_length,
                        self.max_word_length)


        self.char_X_dev = out['char_X']
        self.word_length_X_dev = out['word_length_X']
        self.cap_X_dev = out['cap_X']
        self.word_X_dev = out['word_X']
        self.mask_X_dev = out['mask_X']
        self.sentence_length_X_dev = out['sentence_length_X']
        self.Y_dev = out['Y']

        # Loads the test set
        print "INFO: Reading test data!"
        docs = ut.load_dataset(self.test_set_path)
        Char_X_test, Cap_X_test, Word_X_test, Y_test = ut.docs_to_sentences(
                                                            docs,
                                                            self.word_to_num,
                                                            self.tag_to_num,
                                                            self.char_to_num)

        out = ut.padding(
                        Char_X_test,
                        Cap_X_test,
                        Word_X_test,
                        Y_test,
                        self.max_sentence_length,
                        self.max_word_length)

        self.char_X_test = out['char_X']
        self.word_length_X_test = out['word_length_X']
        self.cap_X_test = out['cap_X']
        self.word_X_test = out['word_X']
        self.mask_X_test = out['mask_X']
        self.sentence_length_X_test = out['sentence_length_X']
        self.Y_test = out['Y']

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.
        """

        self.char_input_placeholder = tf.placeholder(
                                            tf.int32,
                                            shape=(
                                                None,
                                                self.max_sentence_length,
                                                self.max_word_length))

        self.word_length_placeholder = tf.placeholder(
                                            tf.int32,
                                            shape=(None, self.max_sentence_length))

        self.cap_input_placeholder = tf.placeholder(
                                        dtype=tf.int32,
                                        shape=(None, self.max_sentence_length))

        self.word_input_placeholder = tf.placeholder(
                                        dtype=tf.int32,
                                        shape=(None, self.max_sentence_length))

        self.word_mask_placeholder = tf.placeholder(
                                        dtype=tf.float32,
                                        shape=(None, self.max_sentence_length))

        self.sentence_length_placeholder = tf.placeholder(
                                                dtype=tf.int32,
                                                shape=(None,))

        self.tag_placeholder = tf.placeholder(
                                    dtype=tf.int32,
                                    shape=(None, self.max_sentence_length))

        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(
                        self,
                        char_input_batch,
                        word_length_batch,
                        cap_input_batch,
                        word_input_batch,
                        word_mask_batch,
                        sentence_length_batch,
                        dropout_batch,
                        tag_batch=None
                        ):

        """Creates the feed_dict.

        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
            ....
        }
        """

        feed_dict={
            self.char_input_placeholder: char_input_batch,
            self.word_length_placeholder: word_length_batch,
            self.cap_input_placeholder: cap_input_batch,
            self.word_input_placeholder: word_input_batch,
            self.word_mask_placeholder: word_mask_batch,
            self.sentence_length_placeholder: sentence_length_batch,
            self.dropout_placeholder: dropout_batch
            }

        if tag_batch is not None:
            feed_dict[self.tag_placeholder] = tag_batch

        return feed_dict

    def add_embedding(self):
        """Add embedding layer that maps from vocabulary to vectors.
        """

        #Capitilization lookup table
        cap_lookup_table = tf.constant(
                                [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]
                                ],
                                name="cap_lookup_table",
                                dtype=tf.float32)

        cap_embeddings = tf.nn.embedding_lookup(
                                cap_lookup_table,
                                self.cap_input_placeholder)

        word_lookup_table = tf.Variable(
                               	self.word_vectors,
                                name="word_lookup_table",
                                dtype=tf.float32)

        word_embeddings = tf.nn.embedding_lookup(
                                word_lookup_table,
                                self.word_input_placeholder)


        boundry = np.sqrt(np.divide(3.0, self.char_embedding_size))
        new_char_vectors = np.random.uniform(
                                low=-boundry,
                                high=boundry,
                                size=(
                                    len(self.char_to_num),
                                    self.char_embedding_size))

        character_lookup_table = tf.Variable(
                                    new_char_vectors,
                                    name="character_lookup_table",
                                    dtype=tf.float32)

        char_embeddings = tf.nn.embedding_lookup(
                                character_lookup_table,
                                self.char_input_placeholder)

        return char_embeddings, word_embeddings, cap_embeddings

    def xavier_initializer(self, shape, **kargs):
        """Defines an initializer for the Xavier distribution.
        This function will be used as a variable initializer.

        Args:
          shape: Tuple or 1-d array that specifies dimensions of the requested tensor.

        Returns:
          out: tf.Tensor of specified shape sampled from Xavier distribution.
        """

        sum_of_dimensions = tf.reduce_sum(shape)
        epsilon = tf.cast(
                        tf.sqrt( tf.divide(6, sum_of_dimensions) ),
                        tf.float32)

        out = tf.random_uniform(shape,
                                minval=-epsilon,
                                maxval=epsilon,
                                dtype=tf.float32)

        return out

    def add_model(self, char_embeddings, word_embeddings, cap_embeddings):

        #current batch_size
        b_size = tf.shape(word_embeddings)[0]

        char_embeddings_t = tf.reshape(char_embeddings,
                            [-1, self.max_word_length, self.char_embedding_size])

        #character-level bidirectional rnns
        forward_char_level_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=self.char_hidden_units,
                                        use_peepholes=False,
                                        cell_clip=None,
                                        initializer=self.xavier_initializer,
                                        num_proj=None,
                                        proj_clip=None,
                                        num_unit_shards=None,
                                        num_proj_shards=None,
                                        forget_bias=1.0,
                                        state_is_tuple=True,
                                        activation=tf.tanh
                                        )

        backward_char_level_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=self.char_hidden_units,
                                        use_peepholes=False,
                                        cell_clip=None,
                                        initializer=self.xavier_initializer,
                                        num_proj=None,
                                        proj_clip=None,
                                        num_unit_shards=None,
                                        num_proj_shards=None,
                                        forget_bias=1.0,
                                        state_is_tuple=True,
                                        activation=tf.tanh
                                        )

        sequence_length_reshaped = tf.reshape(
                                    self.word_length_placeholder,
                                    (b_size * self.max_sentence_length,))

        (char_h_fw, char_h_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                    forward_char_level_lstm,
                                    backward_char_level_lstm,
                                    char_embeddings_t,
                                    sequence_length=sequence_length_reshaped,
                                    initial_state_fw=None,
                                    initial_state_bw=None,
                                    dtype=tf.float32,
                                    parallel_iterations=None,
                                    swap_memory=False,
                                    time_major=False,
                                    scope="char"
                                    )

        #select last outputs.
        b_index = tf.reshape(
                    tf.multiply(
                    tf.range(0, b_size), self.max_word_length * self.max_sentence_length),
                    (b_size, 1))

        s_index = tf.reshape(
                    tf.multiply(
                    tf.range(0, self.max_sentence_length), self.max_word_length),
                    (1, self.max_sentence_length))

        index = tf.add(
                    tf.add(b_index, s_index),
                    tf.subtract(self.word_length_placeholder, 1))

        fwd_char = tf.gather(
                    tf.reshape(
                        char_h_fw,
                        [b_size * self.max_sentence_length * self.max_word_length, self.char_hidden_units]),
                    index)

        #select first outputs.
        index = tf.add(b_index, s_index)
        bck_char = tf.gather(
                    tf.reshape(
                        char_h_bw,
                        [b_size * self.max_sentence_length * self.max_word_length, self.char_hidden_units]),
                    index)

        """ combine character-level embeddings with word-level embeddings"""
        char_final_embeddings = tf.concat([fwd_char, bck_char], 2)

        t = tf.concat([char_final_embeddings, word_embeddings], 2)
        final_embeddings = tf.nn.dropout(t, self.dropout_placeholder)

	#consider capatilization patterns.
        final_embeddings = tf.concat([final_embeddings, cap_embeddings], axis=2)

        #Creating context embeddings with respect to the windows size = 5
        temp = []
        zeros = tf.zeros((b_size, self.word_embedding_size +  2 * self.char_hidden_units + 4), dtype=tf.float32)
        final_embeddings_t = tf.transpose(final_embeddings, [1,0,2])
        for time_index in range(self.max_sentence_length):
            if time_index == 0:
                f1 = zeros
                f2 = zeros
                f3 = final_embeddings_t[time_index]
                f4 = final_embeddings_t[time_index + 1]
                f5 = final_embeddings_t[time_index + 2]
                new = tf.concat([f1, f2, f3, f4, f5], axis=1)
                temp.append(new)

            elif time_index == 1:
                f1 = zeros
                f2 = final_embeddings_t[time_index - 1]
                f3 = final_embeddings_t[time_index]
                f4 = final_embeddings_t[time_index + 1]
                f5 = final_embeddings_t[time_index + 2]
                new = tf.concat([f1, f2, f3, f4, f5], axis=1)
                temp.append(new)

            elif time_index == (self.max_sentence_length - 2):
                f1 = final_embeddings_t[time_index - 2]
                f2 = final_embeddings_t[time_index - 1]
                f3 = final_embeddings_t[time_index]
                f4 = final_embeddings_t[time_index + 1]
                f5 = zeros
                new = tf.concat([f1, f2, f3, f4, f5], axis=1)
                temp.append(new)

            elif time_index == (self.max_sentence_length - 1):
                f1 = final_embeddings_t[time_index - 2]
                f2 = final_embeddings_t[time_index - 1]
                f3 = final_embeddings_t[time_index]
                f4 = zeros
                f5 = zeros
                new = tf.concat([f1, f2, f3, f4, f5], axis=1)
                temp.append(new)

            else:
                f1 = final_embeddings_t[time_index - 2]
                f2 = final_embeddings_t[time_index - 1]
                f3 = final_embeddings_t[time_index]
                f4 = final_embeddings_t[time_index + 1]
                f5 = final_embeddings_t[time_index + 2]
                new = tf.concat([f1, f2, f3, f4, f5], axis=1)
                temp.append(new)

        temp = tf.stack(temp, axis=1)

        
        forward_word_level_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=self.word_hidden_units,
                                        use_peepholes=False,
                                        cell_clip=None,
                                        initializer=self.xavier_initializer,
                                        num_proj=None,
                                        proj_clip=None,
                                        num_unit_shards=None,
                                        num_proj_shards=None,
                                        forget_bias=1.0,
                                        state_is_tuple=True,
                                        activation=tf.tanh
                                        )

        backward_word_level_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=self.word_hidden_units,
                                        use_peepholes=False,
                                        cell_clip=None,
                                        initializer=self.xavier_initializer,
                                        num_proj=None,
                                        proj_clip=None,
                                        num_unit_shards=None,
                                        num_proj_shards=None,
                                       	forget_bias=1.0,
                                        state_is_tuple=True,
                                        activation=tf.tanh
                                        )


        (h_fw, h_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                    forward_word_level_lstm,
                                    backward_word_level_lstm,
                                    temp,
                                    sequence_length=self.sentence_length_placeholder,
                                    initial_state_fw=None,
                                    initial_state_bw=None,
                                    dtype=tf.float32,
                                    parallel_iterations=None,
                                    swap_memory=False,
                                    time_major=False,
                                    scope="word"
                                    )

        encoder_final_hs = tf.concat([h_fw, h_bw], axis=2)

        #apply dropout
        dropped_encoder_final_hs = tf.nn.dropout(
                                        encoder_final_hs,
                                        self.dropout_placeholder
                                    )

        """hidden layer"""
        with tf.variable_scope("hidden"):
            U_hidden = tf.get_variable(
                            "U_hidden",
                            (2 * self.word_hidden_units, self.word_hidden_units),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_hidden = tf.get_variable(
                            "b_hidden",
                            (self.word_hidden_units,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )


            outputs = tf.add(
                        tf.matmul(
                            tf.reshape(
                                dropped_encoder_final_hs,
                                (-1, 2  *  self.word_hidden_units)
                            ),
                            U_hidden
                        ),
                        b_hidden
                    )
            outputs = tf.tanh(outputs)

        """baseline prediction layer"""
        with tf.variable_scope("baseline"):
       		U_baseline = tf.get_variable(
                            "U_baseline",
                            (self.word_hidden_units, self.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

        	b_baseline = tf.get_variable(
                            "b_baseline",
                            (self.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )


        	preds = tf.add(
                        tf.matmul(
                            tf.reshape(
                                outputs,
                                (-1, self.word_hidden_units)
                            ),
                            U_baseline
                        ),
                        b_baseline
                    )

         	self.preds = tf.reshape(
                        preds,
                        (b_size, self.max_sentence_length, self.tag_size)
                    )

        if self.CRF:
            #CRF
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                                                            self.preds,
                                                            self.tag_placeholder,
                                                            self.sentence_length_placeholder
                                                            )
            return self.log_likelihood

        else:
            return self.preds

    def add_loss_op(self, output_before_softmax):
        """Adds cross_entropy_loss ops to the computational graph.
           OR CRF loss.
        """

        log_likelihood = output_before_softmax
        if self.CRF:
            loss = tf.reduce_mean(-log_likelihood)

        else:
            loss = tf.contrib.seq2seq.sequence_loss(
                                    logits=output_before_softmax,
                                    targets=self.tag_placeholder,
                                    weights=self.word_mask_placeholder,
                                    average_across_timesteps=True,
                                    average_across_batch=True
                                    )
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
          loss: Loss tensor, from cross_entropy_loss.

        Returns:
          train_op: The Op for training.
        """

        #we use adam optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        train_operation = optimizer.apply_gradients(zip(clipped_gradients, variables))

        return train_operation

    def __init__(self):
        """Constructs the network using the helper functions defined above."""
        self.load_data()
        self.add_placeholders()
        char_embed, word_embed, cap_embed = self.add_embedding()
        y = self.add_model(char_embed, word_embed, cap_embed)
        self.loss = self.add_loss_op(y)
        self.train_op = self.add_training_op(self.loss)
        if not self.CRF:
            self.predictions = tf.nn.softmax(y)

    def run_epoch(
                self,
                session,
                char_X,
                word_length_X,
                cap_X,
                word_X,
                mask_X,
                sentence_length_X,
                Y=None,
                verbose=True
                ):

        orig_cap_X, orig_char_X, orig_word_length_X = cap_X, char_X, word_length_X
        orig_word_X, orig_mask_X  = word_X, mask_X
        orig_sentence_length_X, orig_Y = sentence_length_X, Y

        # We're interested in keeping track of the loss during training
        total_loss = []
        total_steps = int(np.ceil(len(word_X) / float(self.batch_size)))
        data = ut.data_iterator(
                    orig_char_X,
                    orig_word_length_X,
                    orig_cap_X,
                    orig_word_X,
                    orig_mask_X,
                    orig_sentence_length_X,
                    self.batch_size,
                    self.tag_size,
                    orig_Y,
		    True)

        for step, (
                    char_input_data,
                    word_length_data,
                    cap_input_data,
                    word_input_data,
                    word_mask_data,
                    sentence_length_data,
                    tag_data) in enumerate(data):

            feed = self.create_feed_dict(
                        char_input_batch=char_input_data,
                        word_length_batch=word_length_data,
                        cap_input_batch=cap_input_data,
                        word_input_batch=word_input_data,
                        word_mask_batch=word_mask_data,
                        sentence_length_batch=sentence_length_data,
                        dropout_batch=self.dropout,
                        tag_batch=tag_data
                    )

            if self.CRF:
                loss, _ = session.run(
                                [self.loss, self.train_op],
                                feed_dict=feed
                                )
            else:
                loss, _, _ = session.run(
                                [self.loss, self.train_op, self.predictions],
                                feed_dict=feed
                                )

            total_loss.append(loss)

            ##
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                                                            step,
                                                            total_steps,
                                                            np.mean(total_loss)
                                                            )
                                )
                sys.stdout.flush()
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        return np.mean(total_loss)

    def predict(
            self,
            session,
            char_X,
            word_length_X,
            cap_X,
            word_X,
            mask_X,
            sentence_length_X,
            Y=None):

        """Make predictions from the provided model."""

        # If Y is given, the loss is also calculated
        # We deactivate dropout by setting it to 1
        dp = 1.0

        losses = []
        results = []
        if np.any(Y):
            data = ut.data_iterator(
                        char_X,
                        word_length_X,
                        cap_X,
                        word_X,
                        mask_X,
                        sentence_length_X,
                        self.batch_size,
                        self.tag_size,
                        Y,
			False)
        else:
            data = ut.data_iterator(
                        char_X,
                        word_length_X,
                        cap_X,
                        word_X,
                        mask_X,
                        sentence_length_X,
                        self.batch_size,
                        self.tag_size,
                        None,
			False)

        for step, (
                    char_input_data,
                    word_length_data,
                    cap_input_data,
                    word_input_data,
                    word_mask_data,
                    sentence_length_data,
                    tag_data) in enumerate(data):

            feed = self.create_feed_dict(
                        char_input_batch=char_input_data,
                        word_length_batch=word_length_data,
                        cap_input_batch=cap_input_data,
                        word_input_batch=word_input_data,
                        word_mask_batch=word_mask_data,
                        sentence_length_batch=sentence_length_data,
                        dropout_batch=dp,
                        tag_batch=tag_data
                    )
            if self.CRF:
                if np.any(tag_data):
                    feed[self.tag_placeholder] = tag_data
                    loss, unary_scores, sequence_lengths, transition_params = session.run(
                                                                        [
                                                                        self.loss,
                                                                        self.preds,
                                                                        self.sentence_length_placeholder,
                                                                        self.transition_params
                                                                        ],
									feed_dict=feed
                                                                     )
                    losses.append(loss)
                else:
                    unary_scores, sequence_lengths, transition_params = session.run(
                                                                        [
                                                                        self.preds,
                                                                        self.sentence_length_placeholder,
                                                                        self.transition_params
                                                                        ],
									feed_dict=feed
                                                                     )
                inner_results = []
                for unary_scores_, sequence_length_ in zip(unary_scores, sequence_lengths):
                    # Remove padding.
                    unary_scores_ = unary_scores_[:sequence_length_]
                    # Compute the highest score and its tag sequence.
                    viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                                                unary_scores_,
                                                                transition_params
                                                            )
                    predicted_indices = viterbi_sequence
                    inner_results.append(predicted_indices)

                results.append(inner_results)
            else:

                if np.any(tag_data):
                    feed[self.tag_placeholder] = tag_data
                    loss, preds = session.run(
                                        [self.loss, self.predictions],
                                        feed_dict=feed
                                        )
                    losses.append(loss)
                else:
                    preds = session.run(self.predictions, feed_dict=feed)


                predicted_indices = preds.argmax(axis=2)
                results.append(predicted_indices)

        if len(losses)==0:
            return 0, results

        else:
            return np.mean(losses), results

    def save_predictions(
                self,
                predictions,
                sentence_length,
                filename,
                words,
                true_tags
                ):

        """Saves predictions to the provided file."""
        with open(filename, "wb") as f:
            for batch_index in range(len(predictions)):
		preds = ut.convert_to_iob(predictions[batch_index], self.num_to_tag, self.tag_to_num)
                batch_predictions = np.array(preds)
                b_size = batch_predictions.shape[0]
                for sentence_index in range(b_size):
                    for word_index in range(self.max_sentence_length):
                        ad = (batch_index * self.batch_size) + sentence_index
                        if(word_index < sentence_length[ad]):
                            to_file = str(self.num_to_word[words[ad][word_index]])
                            to_file += " "
                            to_file += str(self.num_to_tag[true_tags[ad][word_index]])
                            to_file += " "
                            to_file += str(self.num_to_tag[batch_predictions[sentence_index][word_index]])
                            to_file += "\n"
                            f.write(to_file)

                    f.write("\n")

    def eval_fscore(self):
	os.system("%s < %s > %s" % ('./conlleval', 'temp.predicted', 'temp.score'))
	result_lines = [line.rstrip() for line in codecs.open('temp.score', 'r', 'utf8')]
	return float(result_lines[1].strip().split()[-1])

def run_NER():
    """run NER model implementation.
    """

    with tf.Graph().as_default():
        model = NER()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            best_val_loss = float('inf')
            best_val_epoch = 0

            session.run(init)
            first_start = time.time()

            for epoch in xrange(model.max_epochs):
                print
                print 'Epoch {}'.format(epoch)
                start = time.time()
                ###
                train_loss = model.run_epoch(
                                        session,
                                        model.char_X_train,
                                        model.word_length_X_train,
                                        model.cap_X_train,
                                        model.word_X_train,
                                        model.mask_X_train,
                                        model.sentence_length_X_train,
                                        model.Y_train
                                        )
                val_loss , predictions = model.predict(
                                        session,
                                        model.char_X_dev,
                                        model.word_length_X_dev,
                                        model.cap_X_dev,
                                        model.word_X_dev,
                                        model.mask_X_dev,
                                        model.sentence_length_X_dev,
                                        model.Y_dev
                                        )

                print 'Training loss: {}'.format(train_loss)
                print 'Validation loss: {}'.format(val_loss)

		model.save_predictions(
                                predictions,
                                model.sentence_length_X_dev,
                                "temp.predicted",
                                model.word_X_dev,
                                model.Y_dev
                                )

		val_fscore = model.eval_fscore()

		val_fscore_loss = 100.0 - val_fscore

		print 'Validation fscore: {}'.format(val_fscore)
		print 'Validation fscore loss: {}'.format(val_fscore_loss)

                if  val_fscore_loss + val_loss < best_val_loss:
                    best_val_loss = val_loss + val_fscore_loss
                    best_val_epoch = epoch
                    if not os.path.exists("./weights"):
                        os.makedirs("./weights")
                    saver.save(session, './weights/ner.weights')

                # For early stopping which is kind of regularization for network.
                if epoch - best_val_epoch > model.early_stopping:
			break
                ###

                print 'Epoch training time: {} seconds'.format(time.time() - start)

            print 'Total training time: {} seconds'.format(time.time() - first_start)


            saver.restore(session, './weights/ner.weights')
	    print
            print
            print 'Dev'
            start = time.time()
            dev_loss, predictions = model.predict(
                                                session,
                                                model.char_X_dev,
                                                model.word_length_X_dev,
                                                model.cap_X_dev,
                                                model.word_X_dev,
                                                model.mask_X_dev,
                                                model.sentence_length_X_dev,
                                                model.Y_dev
                                                )

            print 'Dev loss: {}'.format(dev_loss)
            print 'Total test time: {} seconds'.format(time.time() - start)
            print 'Writing predictions to dev.predicted'
	    model.save_predictions(
                                predictions,
                                model.sentence_length_X_dev,
                                "dev.predicted",
                                model.word_X_dev,
                                model.Y_dev
                                )

            print
            print
            print 'Test'
            start = time.time()
            test_loss, predictions = model.predict(
                                                session,
                                                model.char_X_test,
                                                model.word_length_X_test,
                                                model.cap_X_test,
                                                model.word_X_test,
                                                model.mask_X_test,
                                                model.sentence_length_X_test,
                                                model.Y_test
                                                )

            print 'Test loss: {}'.format(test_loss)
            print 'Total test time: {} seconds'.format(time.time() - start)
            print 'Writing predictions to test.predicted'
            model.save_predictions(
                                predictions,
                                model.sentence_length_X_test,
                                "test.predicted",
                                model.word_X_test,
                                model.Y_test
                                )

if __name__ == "__main__":
  run_NER()
