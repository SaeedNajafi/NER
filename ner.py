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
    """ Implements an NER (Named Entity Recognition) model """

    """ Model hyperparams and data information """
    word_embedding_size = 100
    char_embedding_size = 25
    word_rnn_hidden_units = 100
    char_rnn_hidden_units = 25

    max_sentence_length = 150
    max_word_length = 25
    tag_size = 17

    batch_size = 16
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 24
    early_stopping = 2

    """inference type"""
    #inference = "softmax"
    #inference = "crf"
    inference = "decoder_rnn"

    """for decoder_rnn"""
    #decoding="greedy"
    decoding="beamsearch"
    beamsize=4
    #decoding="viterbi"


    """path to different files"""
    word_dic_path = './data/glove_en_word_100_dic.txt'
    word_vectors_path = './data/glove_en_word_100_vectors.txt'
    char_dic_path = './data/en_char_dic.txt'
    train_set_path = './data/eng.train.v1'
    dev_set_path = './data/eng.testa.v1'
    test_set_path = './data/eng.testb.v1'

    def __init__(self):
        """Constructs the network using the helper functions defined below."""
        self.load_data()
        self.add_placeholders()
        char_embed, word_embed, cap_embed = self.add_embedding()
        H = self.encoder(char_embed, word_embed, cap_embed)

        if self.inference=="softmax":
            loss = self.train_by_softmax(H)

        elif self.inference=="crf":
            loss = self.train_by_crf(H)

        elif self.inference=="decoder_rnn":
            loss = self.train_by_decoder_rnn(H)

            if self.decoding=="greedy":
                self.greedy_decoding(H)

            elif self.decoding=="beamsearch":
                self.beamsearch_decoding(H, self.beamsize)

            elif self.decoding=="viterbi":
                self.beamsearch_decoding(H, self.tag_size)

        self.train_op = self.add_training_op(loss)

        return

    def load_data(self):
        """
        Loads starter word vectors, creates random character vectors,
        and loads train/dev/test data.

        """

        #Loads the starter word vectors
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
                                                self.max_word_length)
                                            )

        self.word_length_placeholder = tf.placeholder(
                                            tf.int32,
                                            shape=(None, self.max_sentence_length)
                                            )

        self.cap_input_placeholder = tf.placeholder(
                                        dtype=tf.int32,
                                        shape=(None, self.max_sentence_length)
                                        )

        self.word_input_placeholder = tf.placeholder(
                                        dtype=tf.int32,
                                        shape=(None, self.max_sentence_length)
                                        )

        self.word_mask_placeholder = tf.placeholder(
                                        dtype=tf.float32,
                                        shape=(None, self.max_sentence_length)
                                        )

        self.sentence_length_placeholder = tf.placeholder(
                                                dtype=tf.int32,
                                                shape=(None,)
                                                )

        self.tag_placeholder = tf.placeholder(
                                    dtype=tf.int32,
                                    shape=(None, self.max_sentence_length)
                                    )

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

    def diag_initializer(self, shape, **kargs):
        out = tf.diag(tf.ones((self.tag_size,),dtype=tf.float32))
        return out

    def encoder(self, char_embeddings, word_embeddings, cap_embeddings):

        #current batch_size
        b_size = tf.shape(word_embeddings)[0]




        ################################## prefix and suffix information extracting ##################################
        char_embeddings_t = tf.reshape(char_embeddings,
                            [-1, self.max_word_length, self.char_embedding_size])

        #character-level forward lstm cell
        forward_char_level_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=self.char_rnn_hidden_units,
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

        #character-level backward lstm cell
        backward_char_level_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=self.char_rnn_hidden_units,
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

        #character-level bidirectional rnn to
        #construct prefix and suffix information for each word.
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

        #select last outputs for suffix information.

        #batch index
        b_index = tf.reshape(
                        tf.multiply(
                            tf.range(0, b_size),
                            self.max_word_length * self.max_sentence_length
                            ),
                        (b_size, 1)
                        )

        #sentence index
        s_index = tf.reshape(
                        tf.multiply(
                            tf.range(0, self.max_sentence_length),
                            self.max_word_length
                            ),
                        (1, self.max_sentence_length)
                        )

        #index for last character of each word
        index = tf.add(
                    tf.add(b_index, s_index),
                    tf.subtract(self.word_length_placeholder, 1)
                    )

        #select last character's hidden state as suffix information for each word
        fwd_char = tf.gather(
                    tf.reshape(
                        char_h_fw,
                        [b_size * self.max_sentence_length * self.max_word_length,
                        self.char_rnn_hidden_units]
                    ),
                    index
                    )




        #select first outputs for prefix information
        #index for first character of each word
        index = tf.add(b_index, s_index)

        #select first character's hidden state as prefix information for each word
        bck_char = tf.gather(
                    tf.reshape(
                        char_h_bw,
                        [b_size * self.max_sentence_length * self.max_word_length,
                         self.char_rnn_hidden_units]
                    ),
                    index
                    )
        ##################################     End    ##################################


        """ concat prefix/suffix information with word-level embeddings"""
        char_final_embeddings = tf.concat([fwd_char, bck_char], 2)
        t = tf.concat([char_final_embeddings, word_embeddings], 2)
        final_embeddings = tf.nn.dropout(t, self.dropout_placeholder)

        #consider capatilization patterns.
        final_embeddings = tf.concat([final_embeddings, cap_embeddings], axis=2)

        #Creating context with respect to the window size = 5
        temp = []
        zeros = tf.zeros(
                (b_size, self.word_embedding_size +  2 * self.char_rnn_hidden_units + 4),
                dtype=tf.float32
                )
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


        ##################################     word-level encoder    ##################################
        forward_word_level_lstm = tf.contrib.rnn.LSTMCell(
                                        num_units=self.word_rnn_hidden_units,
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
                                        num_units=self.word_rnn_hidden_units,
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

        ##################################     End   ##################################

        """hidden layer"""
        with tf.variable_scope("hidden"):
            U_hidden = tf.get_variable(
                            "U_hidden",
                            (2 * self.word_rnn_hidden_units, self.word_rnn_hidden_units),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_hidden = tf.get_variable(
                            "b_hidden",
                            (self.word_rnn_hidden_units,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )


            H = tf.add(
                        tf.matmul(
                            tf.reshape(
                                dropped_encoder_final_hs,
                                (-1, 2  *  self.word_rnn_hidden_units)
                            ),
                            U_hidden
                        ),
                        b_hidden
                    )
            H = tf.tanh(H)
            H = tf.reshape(H, (-1, self.max_sentence_length, self.word_rnn_hidden_units))

        return H

    def train_by_softmax(self, H):

        """
        Apply a softmax layer to get a probability for each tag.
        Define the loss during training and do testing to save predictions.
        """

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            U_softmax = tf.get_variable(
                            "U_softmax",
                            (self.word_rnn_hidden_units, self.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (self.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

            preds = tf.add(
                        tf.matmul(
                            tf.reshape(
                                H,
                                (-1, self.word_rnn_hidden_units)
                            ),
                            U_softmax
                        ),
                        b_softmax
                    )

            self.preds = tf.reshape(
                        preds,
                        (-1, self.max_sentence_length, self.tag_size)
                        )


        self.loss = tf.contrib.seq2seq.sequence_loss(
                                    logits=self.preds,
                                    targets=self.tag_placeholder,
                                    weights=self.word_mask_placeholder,
                                    average_across_timesteps=True,
                                    average_across_batch=True
                                    )

        self.predictions = tf.nn.softmax(self.preds)
        return self.loss

    def train_by_crf(self, H):

        """
        Apply a crf layer to get a probability for each tag.
        Define the loss during training.
        """

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            U_softmax = tf.get_variable(
                            "U_softmax",
                            (self.word_rnn_hidden_units, self.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (self.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )


            preds = tf.add(
                        tf.matmul(
                            tf.reshape(
                                H,
                                (-1, self.word_rnn_hidden_units)
                            ),
                            U_softmax
                        ),
                        b_softmax
                    )

            self.preds = tf.reshape(
                        preds,
                        (-1, self.max_sentence_length, self.tag_size)
                        )

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                                                            self.preds,
                                                            self.tag_placeholder,
                                                            self.sentence_length_placeholder
                                                            )

        self.loss = tf.reduce_mean(-self.log_likelihood)

        return self.loss

    def train_by_decoder_rnn(self, H):

        """
        Apply a decoder_rnn layer in the training step to get a score for each tag.
        Defines the loss during training.
        """

        #we need to define a tag embedding layer.
        with tf.variable_scope("tag_embedding_layer"):
            tag_lookup_table = tf.get_variable(
                                    name = "tag_lookup_table",
                                    shape = (self.tag_size, self.tag_size),
                                    dtype= tf.float32,
                                    trainable= True,
                                    initializer = self.diag_initializer
                                    )

        tag_embeddings = tf.nn.embedding_lookup(tag_lookup_table, self.tag_placeholder)
        b_size = tf.shape(tag_embeddings)[0]

        #add GO symbol into the begining of every sentence and
        #shift rest by one position.

        temp = []
        GO_symbol = tf.zeros((b_size, self.tag_size), dtype=tf.float32)
        tag_embeddings_t = tf.transpose(tag_embeddings, [1,0,2])
        for time_index in range(self.max_sentence_length):
            if time_index==0:
                temp.append(GO_symbol)
            else:
                temp.append(tag_embeddings_t[time_index-1])

        temp = tf.stack(temp, axis=1)

        tag_embeddings_final = temp
        with tf.variable_scope('decoder_rnn') as scope:
            self.decoder_lstm_cell = tf.contrib.rnn.LSTMCell(
                                        num_units=self.tag_size,
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

            tag_scores, _ = tf.nn.dynamic_rnn(
                                    self.decoder_lstm_cell,
                                    tag_embeddings_final,
                                    sequence_length=self.sentence_length_placeholder,
                                    initial_state=None,
                                    dtype=tf.float32,
                                    parallel_iterations=None,
                                    swap_memory=False,
                                    time_major=False,
                                    scope=scope
                                    )

        tag_scores_dropped = tf.nn.dropout(tag_scores, self.dropout_placeholder)
        H_and_tag_scores = tf.concat([H,tag_scores_dropped], axis=2)

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            U_softmax = tf.get_variable(
                            "U_softmax",
                            (self.word_rnn_hidden_units + self.tag_size, self.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (self.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

            preds = tf.add(
                        tf.matmul(
                            tf.reshape(
                                H_and_tag_scores,
                                (-1, self.word_rnn_hidden_units + self.tag_size)
                            ),
                            U_softmax
                        ),
                        b_softmax
                    )

            self.preds = tf.reshape(
                            preds,
                            (-1, self.max_sentence_length, self.tag_size)
                        )


        self.loss = tf.contrib.seq2seq.sequence_loss(
                                    logits=self.preds,
                                    targets=self.tag_placeholder,
                                    weights=self.word_mask_placeholder,
                                    average_across_timesteps=True,
                                    average_across_batch=True
                                    )

        return self.loss

    def greedy_decoding(self, H):

        #batch size
        b_size = tf.shape(H)[0]

        """Reload softmax prediction layer"""
        with tf.variable_scope("softmax", reuse=True):
            U_softmax = tf.get_variable("U_softmax")
            b_softmax = tf.get_variable("b_softmax")

        #reloading the tag embedding layer.
        with tf.variable_scope("tag_embedding_layer", reuse=True):
            tag_lookup_table = tf.get_variable("tag_lookup_table")



        GO_symbol = tf.zeros((b_size, self.tag_size), dtype=tf.float32)
        initial_state = self.decoder_lstm_cell.zero_state(b_size, tf.float32)
        H_t = tf.transpose(H, [1,0,2])
        outputs = []
        with tf.variable_scope("decoder_rnn", reuse=True) as scope:
            for time_index in range(self.max_sentence_length):
                if time_index==0:
                    output, state = self.decoder_lstm_cell(GO_symbol, initial_state, scope)
                else:
                    prev_output = tf.nn.embedding_lookup(tag_lookup_table, predicted_indices)
                    output, state = self.decoder_lstm_cell(prev_output, state, scope)


                H_and_output = tf.concat([H_t[time_index], output], axis=1)

                pred = tf.add(tf.matmul(H_and_output, U_softmax), b_softmax)
                predictions = tf.nn.softmax(pred)
                predicted_indices = tf.argmax(predictions, axis=1)
                outputs.append(predicted_indices)

            self.outputs = tf.stack(outputs, axis=1)

        return

    def beamsearch_decoding(self, H, beamsize):

        #batch size
        b_size = tf.shape(H)[0]

        """Reload softmax prediction layer"""
        with tf.variable_scope("softmax", reuse=True):
            U_softmax = tf.get_variable("U_softmax")
            b_softmax = tf.get_variable("b_softmax")

        #we need to reload the tag embedding layer.
        with tf.variable_scope("tag_embedding_layer", reuse=True):
            tag_lookup_table = tf.get_variable("tag_lookup_table")

        GO_symbol = tf.zeros((b_size, self.tag_size), dtype=tf.float32)
        initial_state = self.decoder_lstm_cell.zero_state(b_size, tf.float32)
        H_t = tf.transpose(H, [1,0,2])

        with tf.variable_scope("decoder_rnn", reuse=True) as scope:
            for time_index in range(self.max_sentence_length):
                if time_index==0:
                    output, (c_state, m_state) = self.decoder_lstm_cell(GO_symbol, initial_state, scope)
                    H_and_output = tf.concat([H_t[time_index], output], axis=1)
                    pred = tf.add(tf.matmul(H_and_output, U_softmax), b_softmax)
                    predictions = tf.nn.softmax(pred)
                    probs, indices = tf.nn.top_k(predictions, k=beamsize, sorted=True)
                    prev_indices = indices
                    beam = tf.expand_dims(indices, axis=2)
                    prev_probs = tf.log(probs)
                    prev_c_states = [c_state for i in range(beamsize)]
                    prev_c_states = tf.stack(prev_c_states, axis=1)
                    prev_m_states = [m_state for i in range(beamsize)]
                    prev_m_states = tf.stack(prev_m_states, axis=1)

                else:
                    prev_indices_t = tf.transpose(prev_indices, [1,0])
                    prev_probs_t = tf.transpose(prev_probs, [1,0])
                    prev_c_states_t = tf.transpose(prev_c_states, [1,0,2])
                    prev_m_states_t = tf.transpose(prev_m_states, [1,0,2])

                    beam_t = tf.transpose(beam, [1,0,2])
		    probs_candidates = []
		    indices_candidates = []
		    beam_candidates = []
		    c_state_candidates = []
		    m_state_candidates = []
                    for b in range(beamsize):
                        prev_output = tf.nn.embedding_lookup(tag_lookup_table, prev_indices_t[b])
                        output, (c_state, m_state) = self.decoder_lstm_cell(prev_output, (prev_c_states_t[b],prev_m_states_t[b]), scope)

                        H_and_output = tf.concat([H_t[time_index], output], axis=1)
                        pred = tf.add(tf.matmul(H_and_output, U_softmax), b_softmax)
                        predictions = tf.nn.softmax(pred)
                        probs, indices = tf.nn.top_k(predictions, k=beamsize, sorted=True)
                        probs_t = tf.transpose(probs, [1,0])
                        indices_t = tf.transpose(indices, [1,0])
                        for bb in range(beamsize):
                            probs_candidates.append(tf.add(prev_probs_t[b], tf.log(probs_t[bb])))
                            indices_candidates.append(indices_t[bb])
                            beam_candidates.append(tf.concat([beam_t[b], tf.expand_dims(indices_t[bb], axis=1)], axis=1))
                            c_state_candidates.append(c_state)
                            m_state_candidates.append(m_state)

                    temp_probs = tf.stack(probs_candidates, axis=1)
                    temp_indices = tf.stack(indices_candidates, axis=1)
                    temp_beam = tf.stack(beam_candidates, axis=1)
                    temp_c_states = tf.stack(c_state_candidates, axis=1)
                    temp_m_states = tf.stack(m_state_candidates, axis=1)
                    _, max_indices = tf.nn.top_k(temp_probs, k=beamsize, sorted=True)

                    #batch index
                    b_index = tf.reshape(tf.range(0, b_size),(b_size, 1))

                    #beam index
                    be_index = tf.constant(beamsize*beamsize, dtype=tf.int32, shape=(1, beamsize))

                    #index
                    index = tf.add(
                                tf.matmul(b_index, be_index),
                                max_indices
                                )
                    prev_probs = tf.gather(tf.reshape(temp_probs, [-1]), index)
                    prev_indices = tf.gather(tf.reshape(temp_indices, [-1]), index)
                    beam = tf.gather(tf.reshape(temp_beam, [-1, time_index+1]), index)
                    prev_c_states = tf.gather(tf.reshape(temp_c_states, [-1, self.tag_size]), index)
                    prev_m_states = tf.gather(tf.reshape(temp_m_states, [-1, self.tag_size]), index)

            beam_t = tf.transpose(beam, [1,0,2])
            self.outputs = beam_t[0]

        return

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


            loss, _ = session.run([self.loss, self.train_op],feed_dict=feed)

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

            if self.inference=="crf":
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

            elif self.inference=="softmax":

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

            elif self.inference=="decoder_rnn":
                if np.any(tag_data):
                    feed[self.tag_placeholder] = tag_data

                batch_predicted_indices = session.run([self.outputs], feed_dict=feed)

                results.append(batch_predicted_indices[0])

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
                batch_predictions = predictions[batch_index]
                b_size = len(batch_predictions)
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

                _ , predictions = model.predict(
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

                if  val_fscore_loss < best_val_loss:
                    best_val_loss = val_fscore_loss
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
            print 'Total prediction time: {} seconds'.format(time.time() - start)
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
            print 'Total prediction time: {} seconds'.format(time.time() - start)
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
