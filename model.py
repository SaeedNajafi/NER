import tensorflow as tf
import numpy as np

class NER(object):
    """ Implements an NER (Named Entity Recognition) model """
    def __init__(self, config, word_vectors, sd):
        """Constructs the network using the helper functions defined below."""
        self.seed = sd
        self.placeholders(config)
        char_embed, word_embed, cap_embed = self.embeddings(config, word_vectors)

        H = self.encoder(char_embed, word_embed, cap_embed, config)

        if config.inference=="CRF":
            self.train_by_crf(H, config)

        elif config.inference=="R-RNN" or config.inference=="BR-RNN" or config.inference=="RNN" or config.inference=="AC-RNN":
            self.train_by_actor_critic_rnn(H, config)

            with tf.variable_scope("V_adam_optimizer"):
                self.V_train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.V_loss)

            if not config.beamsearch:
                self.outputs = self.greedy_decoding(H, config)

            else:
                self.outputs = self.beam_decoding(H, config)

        elif config.inference=="INDP":
            self.loss, self.outputs = self.train_by_softmax(H, config)

        elif config.inference=="SCH" or config.inference=="DIF-SCH":
            self.train_by_scheduled_decoder_rnn(H, config)

            if not config.beamsearch:
                self.outputs = self.greedy_decoding(H, config)

            else:
                self.outputs = self.beam_decoding(H, config)

        self.train_op = self.add_training_op(self.loss, config)
        return

    def placeholders(self, config):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.
        """

        self.char_input_placeholder = tf.placeholder(
                                            tf.int32,
                                            shape=(
                                                None,
                                                config.max_sentence_length,
                                                config.max_word_length)
                                            )

        self.word_length_placeholder = tf.placeholder(
                                            tf.int32,
                                            shape=(None, config.max_sentence_length)
                                            )

        self.cap_input_placeholder = tf.placeholder(
                                        dtype=tf.int32,
                                        shape=(None, config.max_sentence_length)
                                        )

        self.word_input_placeholder = tf.placeholder(
                                        dtype=tf.int32,
                                        shape=(None, config.max_sentence_length)
                                        )

        self.word_mask_placeholder = tf.placeholder(
                                        dtype=tf.float32,
                                        shape=(None, config.max_sentence_length)
                                        )

        self.sentence_length_placeholder = tf.placeholder(
                                                dtype=tf.int32,
                                                shape=(None,)
                                                )

        self.tag_placeholder = tf.placeholder(
                                    dtype=tf.int32,
                                    shape=(None, config.max_sentence_length)
                                    )

        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, shape=())


        self.alpha_placeholder = tf.placeholder(dtype=tf.float32, shape=())

        self.schedule_placeholder = tf.placeholder(dtype=tf.float32, shape=())

        self.coin_probs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, config.max_sentence_length))

        self.beta_placeholder = tf.placeholder(dtype=tf.float32, shape=())

    def create_feed_dict(
                        self,
                        char_input_batch,
                        word_length_batch,
                        cap_input_batch,
                        word_input_batch,
                        word_mask_batch,
                        sentence_length_batch,
                        dropout_batch,
                        alpha,
                        schedule,
                        coin_probs,
                        beta,
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
            self.dropout_placeholder: dropout_batch,
            self.alpha_placeholder: alpha,
            self.schedule_placeholder: schedule,
            self.coin_probs_placeholder: coin_probs,
            self.beta_placeholder: beta
            }

        if tag_batch is not None:
            feed_dict[self.tag_placeholder] = tag_batch

        return feed_dict

    def embeddings(self, config, word_vectors):
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
                                dtype=tf.float32
                                )

        cap_embeddings = tf.nn.embedding_lookup(
                                cap_lookup_table,
                                self.cap_input_placeholder
                                )

        with tf.variable_scope("word_embeddings"):
            word_lookup_table = tf.Variable(
                               	    word_vectors,
                                    name="word_lookup_table",
                                    dtype=tf.float32
                                    )

        word_embeddings = tf.nn.embedding_lookup(
                                word_lookup_table,
                                self.word_input_placeholder
                                )

        with tf.variable_scope("char_embeddings"):
            character_lookup_table = tf.get_variable(
                                    name = "character_lookup_table",
                                    shape = (config.char_size, config.char_embedding_size),
                                    dtype= tf.float32,
                                    trainable= True,
                                    initializer = self.xavier_initializer
                                    )

        char_embeddings = tf.nn.embedding_lookup(
                                character_lookup_table,
                                self.char_input_placeholder
                                )

        return char_embeddings, word_embeddings, cap_embeddings

    def xavier_initializer (self, shape, **kargs):
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
                        tf.float32
                        )

        out = tf.random_uniform(shape,
                                minval=-epsilon,
                                maxval=epsilon,
                                dtype=tf.float32,
                                seed=self.seed
                                )

        return out

    def encoder(self, char_embeddings, word_embeddings, cap_embeddings, config):

        #current batch_size
        b_size = tf.shape(word_embeddings)[0]




        ################################## prefix and suffix information extracting ##################################
        char_embeddings_t = tf.reshape(char_embeddings,
                            [-1, config.max_word_length, config.char_embedding_size])

        with tf.variable_scope('char_rnn') as scope:

            #character-level forward lstm cell
            forward_char_level_lstm = tf.contrib.rnn.LSTMCell(
                                            num_units=config.char_rnn_hidden_units,
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
                                            num_units=config.char_rnn_hidden_units,
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
                                        (b_size * config.max_sentence_length,)
                                        )

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
                                        scope=scope
                                        )

        #select last outputs for suffix information.

        #batch index
        b_index = tf.reshape(
                        tf.multiply(
                            tf.range(0, b_size),
                            config.max_word_length * config.max_sentence_length
                            ),
                        (b_size, 1)
                        )

        #sentence index
        s_index = tf.reshape(
                        tf.multiply(
                            tf.range(0, config.max_sentence_length),
                            config.max_word_length
                            ),
                        (1, config.max_sentence_length)
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
                        [b_size * config.max_sentence_length * config.max_word_length,
                        config.char_rnn_hidden_units]
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
                        [b_size * config.max_sentence_length * config.max_word_length,
                         config.char_rnn_hidden_units]
                    ),
                    index
                    )
        ##################################     End    ##################################


        """ concat prefix/suffix information with word-level embeddings"""
        char_final_embeddings = tf.concat([fwd_char, bck_char], 2)
        t = tf.concat([char_final_embeddings, word_embeddings], 2)
        final_embeddings = tf.nn.dropout(t, self.dropout_placeholder, seed=self.seed)

        #consider capatilization patterns.
        final_embeddings = tf.concat([final_embeddings, cap_embeddings], axis=2)


        ##################################     word-level encoder    ##################################
        with tf.variable_scope('word_rnn') as scope:

            forward_word_level_lstm = tf.contrib.rnn.LSTMCell(
                                            num_units=config.word_rnn_hidden_units,
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
                                            num_units=config.word_rnn_hidden_units,
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
                                        final_embeddings,
                                        sequence_length=self.sentence_length_placeholder,
                                        initial_state_fw=None,
                                        initial_state_bw=None,
                                        dtype=tf.float32,
                                        parallel_iterations=None,
                                        swap_memory=False,
                                        time_major=False,
                                        scope=scope
                                        )

        encoder_final_hs = tf.concat([h_fw, h_bw], axis=2)

        #apply dropout
        dropped_encoder_final_hs = tf.nn.dropout(
                                        encoder_final_hs,
                                        self.dropout_placeholder,
                                        seed= self.seed
                                    )

        ##################################     End   ##################################

        """hidden layer"""
        with tf.variable_scope("hidden"):
            W_hidden = tf.get_variable(
                            "W_hidden",
                            (2 * config.word_rnn_hidden_units, config.word_rnn_hidden_units),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_hidden = tf.get_variable(
                            "b_hidden",
                            (config.word_rnn_hidden_units,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )


            H = tf.add(
                        tf.matmul(
                            tf.reshape(
                                dropped_encoder_final_hs,
                                (-1, 2  *  config.word_rnn_hidden_units)
                            ),
                            W_hidden
                        ),
                        b_hidden
                    )
            H = tf.tanh(H)
            H = tf.reshape(H, (-1, config.max_sentence_length, config.word_rnn_hidden_units))

        return H

    def train_by_softmax(self, H, config):

        """
        Apply a softmax layer to get a probability for each tag.
        Define the loss during training and do testing to save outputs.
        """

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                            "W_softmax",
                            (config.word_rnn_hidden_units, config.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (config.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

            preds = tf.add(
                        tf.matmul(
                            tf.reshape(
                                H,
                                (-1, config.word_rnn_hidden_units)
                            ),
                            W_softmax
                        ),
                        b_softmax
                    )

            preds = tf.reshape(
                        preds,
                        (-1, config.max_sentence_length, config.tag_size)
                        )


        loss = tf.contrib.seq2seq.sequence_loss(
                                    logits=preds,
                                    targets=self.tag_placeholder,
                                    weights=self.word_mask_placeholder,
                                    average_across_timesteps=True,
                                    average_across_batch=True
                                    )

        predictions = tf.nn.softmax(preds)
        outputs = tf.argmax(predictions, axis=2)
        return loss, outputs

    def train_by_crf(self, H, config):

        """
        Apply a crf layer to get a probability for each tag.
        Define the loss during training.
        """

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                            "W_softmax",
                            (config.word_rnn_hidden_units, config.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (config.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )


            preds = tf.add(
                        tf.matmul(
                            tf.reshape(
                                H,
                                (-1, config.word_rnn_hidden_units)
                            ),
                            W_softmax
                        ),
                        b_softmax
                    )

            self.preds = tf.reshape(
                        preds,
                        (-1, config.max_sentence_length, config.tag_size)
                        )

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                                                            self.preds,
                                                            self.tag_placeholder,
                                                            self.sentence_length_placeholder
                                                            )

        self.loss = tf.reduce_mean(-self.log_likelihood)

        return self.loss


    def train_by_scheduled_decoder_rnn(self, H, config):
        b_size = tf.shape(H)[0]

        #we need to define a tag embedding layer.
        with tf.variable_scope("tag_embedding_layer"):
            tag_lookup_table = tf.get_variable(
                                    name = "tag_lookup_table",
                                    shape = (config.tag_size, config.tag_embedding_size),
                                    dtype= tf.float32,
                                    trainable= True,
                                    initializer = self.xavier_initializer
                                    )

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                            "W_softmax",
                            (config.word_rnn_hidden_units + config.decoder_rnn_hidden_units, config.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (config.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

        self.decoder_lstm_cell = tf.contrib.rnn.LSTMCell(
                                            num_units=config.decoder_rnn_hidden_units,
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

        tag_embeddings = tf.nn.embedding_lookup(tag_lookup_table, self.tag_placeholder)
        GO_symbol = tf.zeros((b_size, config.tag_embedding_size), dtype=tf.float32)
        GO_context = tf.zeros((b_size, config.word_rnn_hidden_units), dtype=tf.float32)

        switch = tf.greater(self.coin_probs_placeholder, self.schedule_placeholder)
        switch = tf.cast(switch, dtype=tf.float32)
        switch_t = tf.transpose(switch, [1,0])

        tag_embeddings_t = tf.transpose(tag_embeddings, [1,0,2])
        H_t = tf.transpose(H, [1,0,2])
        Logits = []
        with tf.variable_scope('decoder_rnn') as scope:
            initial_state = self.decoder_lstm_cell.zero_state(b_size, tf.float32)
            for time_index in range(config.max_sentence_length):
                if time_index==0:
                    inp = tf.concat([GO_symbol, GO_context], axis=1)
                    output, state = self.decoder_lstm_cell(inp, initial_state)
                else:
                    scope.reuse_variables()
                    gold_token = tf.concat([tag_embeddings_t[time_index-1], H_t[time_index-1]], axis=1)
                    beta = self.beta_placeholder
                    if config.inference=='DIF-SCH':
                        prev_output = tf.matmul(tf.nn.softmax(beta * logits), tag_lookup_table)
                    else:
                        prev_output = tf.nn.embedding_lookup(tag_lookup_table, tf.argmax(tf.nn.softmax(logits), axis=1))

                    generated_token = tf.concat([prev_output, H_t[time_index-1]], axis=1)
                    sw = tf.expand_dims(switch_t[time_index-1], axis=1)
                    inp = tf.multiply(generated_token, sw) + tf.multiply(gold_token, 1.0-sw)
                    output, state = self.decoder_lstm_cell(inp, state)

                output_dropped = tf.nn.dropout(output, self.dropout_placeholder, seed=self.seed)
                H_and_output = tf.concat([H_t[time_index], output_dropped], axis=1)
                logits = tf.add(tf.matmul(H_and_output, W_softmax), b_softmax)
                Logits.append(logits)

        Logits = tf.stack(Logits, axis=1)
        cross_loss = tf.contrib.seq2seq.sequence_loss(
                                        logits=Logits,
                                        targets=self.tag_placeholder,
                                        weights=self.word_mask_placeholder,
                                        average_across_timesteps=True,
                                        average_across_batch=True
                                        )
        self.loss = cross_loss
        return self.loss

    def train_by_actor_critic_rnn(self, H, config):
        """
        Apply an actor layer in the training step.
        Defines the loss during training.
        """

        b_size = tf.shape(H)[0]

        #we need to define a tag embedding layer.
        with tf.variable_scope("tag_embedding_layer"):
            tag_lookup_table = tf.get_variable(
                                    name = "tag_lookup_table",
                                    shape = (config.tag_size, config.tag_embedding_size),
                                    dtype= tf.float32,
                                    trainable= True,
                                    initializer = self.xavier_initializer
                                    )

        """softmax prediction layer"""
        with tf.variable_scope("softmax"):
            W_softmax = tf.get_variable(
                            "W_softmax",
                            (config.word_rnn_hidden_units + config.decoder_rnn_hidden_units, config.tag_size),
                            tf.float32,
                            self.xavier_initializer
                            )

            b_softmax = tf.get_variable(
                            "b_softmax",
                            (config.tag_size,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

        with tf.variable_scope("V"):
            W1_V = tf.get_variable(
                                "W1_V",
                                (config.word_rnn_hidden_units + config.decoder_rnn_hidden_units, 64),
                                tf.float32,
                                self.xavier_initializer
                                )

            b1_V = tf.get_variable(
                            "b1_V",
                            (64,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

            W2_V = tf.get_variable(
                            "W2_V",
                            (64, 1),
                            tf.float32,
                            self.xavier_initializer
                            )

            b2_V = tf.get_variable(
                            "b2_V",
                            (1,),
                            tf.float32,
                            tf.constant_initializer(0.0)
                            )

        self.decoder_lstm_cell = tf.contrib.rnn.LSTMCell(
                                            num_units=config.decoder_rnn_hidden_units,
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

        #add GO symbol into the begining of every sentence and
        #shift rest by one position.
        tag_embeddings = tf.nn.embedding_lookup(tag_lookup_table, self.tag_placeholder)
        GO_symbol = tf.zeros((b_size, config.tag_embedding_size), dtype=tf.float32)
        GO_context = tf.zeros((b_size, config.word_rnn_hidden_units), dtype=tf.float32)

        temp = []
        tag_embeddings_t = tf.transpose(tag_embeddings, [1,0,2])
        H_t = tf.transpose(H, [1,0,2])
        for time_index in range(config.max_sentence_length):
            if time_index==0:
                temp.append(tf.concat([GO_symbol, GO_context], axis=1))
            else:
                temp.append(tf.concat([tag_embeddings_t[time_index-1], H_t[time_index-1]], axis=1))

        temp = tf.stack(temp, axis=1)

        embeddings_final = temp

        def maximum_likelihood():
            with tf.variable_scope('decoder_rnn') as scope:
                out_states, _ = tf.nn.dynamic_rnn(
                                        self.decoder_lstm_cell,
                                        embeddings_final,
                                        sequence_length=self.sentence_length_placeholder,
                                        initial_state=None,
                                        dtype=tf.float32,
                                        parallel_iterations=None,
                                        swap_memory=False,
                                        time_major=False,
                                        scope=scope
                                        )

            out_states_dropped = tf.nn.dropout(out_states, self.dropout_placeholder, seed=self.seed)
            H_and_out_states = tf.concat([H, out_states_dropped], axis=2)
            preds = tf.add(
                            tf.matmul(
                                tf.reshape(
                                    H_and_out_states,
                                    (-1, config.word_rnn_hidden_units + config.decoder_rnn_hidden_units)
                                ),
                                W_softmax
                            ),
                            b_softmax
                        )

            Preds = tf.reshape(
                                preds,
                                (-1, config.max_sentence_length, config.tag_size)
                            )

            cross_loss = tf.contrib.seq2seq.sequence_loss(
                                        logits=Preds,
                                        targets=self.tag_placeholder,
                                        weights=self.word_mask_placeholder,
                                        average_across_timesteps=True,
                                        average_across_batch=True
                                        )

            #b2_V is a dummy loss added for coding purposes.
            return cross_loss, b2_V

        def actor_critic():
            Policies = []
            V = []
            with tf.variable_scope('decoder_rnn', reuse=True) as scope:
                initial_state = self.decoder_lstm_cell.zero_state(b_size, tf.float32)
            	for time_index in range(config.max_sentence_length):
                    if time_index==0:
                        inp = tf.concat([GO_symbol, GO_context], axis=1)
                        output, state = self.decoder_lstm_cell(inp, initial_state)
                    else:
                        scope.reuse_variables()
                        inp = tf.concat([prev_output, H_t[time_index-1]], axis=1)
                        output, state = self.decoder_lstm_cell(inp, state)

                    output_dropped = tf.nn.dropout(output, self.dropout_placeholder, seed=self.seed)
                    H_and_output = tf.concat([H_t[time_index], output_dropped], axis=1)

                    #forward pass for the V estimation
                    v = tf.add(tf.matmul(tf.stop_gradient(H_and_output), W1_V), b1_V)
                    v = tf.add(tf.matmul(v, W2_V), b2_V)
                    V.append(v)

                    pred = tf.add(tf.matmul(H_and_output, W_softmax), b_softmax)
                    policy = tf.nn.softmax(pred)
                    Policies.append(policy)

                    prev_output = tf.nn.embedding_lookup(tag_lookup_table, tf.argmax(policy, axis=1))

            Policies = tf.stack(Policies, axis=1)
            V = tf.stack(V, axis=1)

            V = tf.reshape(V, (-1, config.max_sentence_length))

            is_true_tag = tf.cast(tf.equal(tf.cast(self.tag_placeholder, tf.int64), tf.argmax(Policies, axis=2)), tf.float32)
            #hamming loss 0, 1
            Rewards = is_true_tag
            Rewards = tf.multiply(Rewards, self.word_mask_placeholder)
            V = tf.multiply(V, self.word_mask_placeholder)

            Rewards_t = tf.transpose(Rewards, [1,0])
            V_t = tf.transpose(V, [1,0])

            TD_Returns = []
            zeros = tf.cast(self.sentence_length_placeholder - self.sentence_length_placeholder, tf.float32)
            for t in range(config.max_sentence_length):
                ret = zeros
                for i in range(config.n_step):
                    if t + i < config.max_sentence_length:
                        ret += (config.gamma ** i) * Rewards_t[t + i]
                        if i == config.n_step - 1:
                            if t + i + 1 < config.max_sentence_length:
                                ret += (config.gamma ** config.n_step) * V_t[t + config.n_step]
                TD_Returns.append(ret)

            MC_Returns = []
            zeros = tf.cast(self.sentence_length_placeholder - self.sentence_length_placeholder, tf.float32)
            for t in range(config.max_sentence_length):
                ret = zeros
                for i in range(0, config.max_sentence_length - t):
                    ret += (config.gamma ** i) * Rewards_t[t + i]
                MC_Returns.append(ret)

            if config.inference=='AC-RNN':
                Returns = TD_Returns
            else:
                Returns = MC_Returns

            Returns = tf.stack(Returns, axis=1)
            max_policy = tf.reduce_max(Policies, axis=2)

            if config.inference=='R-RNN':
                Objective = tf.log(max_policy) * tf.stop_gradient(Returns)
            else:
                Objective = tf.log(max_policy) * tf.stop_gradient(Returns - V)

            Objective_masked = tf.multiply(Objective, self.word_mask_placeholder)

            V_loss = tf.reduce_mean(tf.pow(tf.stop_gradient(Returns) - V, 2) * self.word_mask_placeholder)
            # L2 regulirization for linear regressor: V
            b = 0.001
            reg1 = tf.nn.l2_loss(W1_V)
            reg2 = tf.nn.l2_loss(W2_V)
            V_loss = b * tf.reduce_mean(reg1) + b * tf.reduce_mean(reg2) + V_loss

            actor_critic_loss = -tf.reduce_mean(Objective_masked)
            return actor_critic_loss, V_loss

        a = self.alpha_placeholder
        def AC_RNN():
            cross_loss, dummy_loss = maximum_likelihood()
            actor_loss, V_loss = actor_critic()
            loss = cross_loss * (1-a) + actor_loss * a
            return loss, V_loss

        def RNN():
            cross_loss, dummy_loss = maximum_likelihood()
            return cross_loss, dummy_loss

        self.loss, self.V_loss = tf.cond(tf.equal(a, tf.constant(0.0, dtype=tf.float32, shape=())), RNN, AC_RNN)
        return self.loss, self.V_loss

    def greedy_decoding(self, H, config):

        #batch size
        b_size = tf.shape(H)[0]

        """Reload softmax prediction layer"""
        with tf.variable_scope("softmax", reuse=True):
            W_softmax = tf.get_variable("W_softmax")
            b_softmax = tf.get_variable("b_softmax")

        #reloading the tag embedding layer.
        with tf.variable_scope("tag_embedding_layer", reuse=True):
            tag_lookup_table = tf.get_variable("tag_lookup_table")

        GO_symbol = tf.zeros((b_size, config.tag_embedding_size), dtype=tf.float32)
        GO_context = tf.zeros((b_size, config.word_rnn_hidden_units), dtype=tf.float32)
        initial_state = self.decoder_lstm_cell.zero_state(b_size, tf.float32)
        H_t = tf.transpose(H, [1,0,2])
        outputs = []

        with tf.variable_scope("decoder_rnn", reuse=True) as scope:
            for time_index in range(config.max_sentence_length):
                if time_index==0:
                    inp = tf.concat([GO_symbol, GO_context], axis=1)
                    output, state = self.decoder_lstm_cell(inp, initial_state)
                else:
                    prev_output = tf.nn.embedding_lookup(tag_lookup_table, predicted_indices)
                    inp = tf.concat([prev_output, H_t[time_index-1]], axis=1)
                    output, state = self.decoder_lstm_cell(inp, state)


                H_and_output = tf.concat([H_t[time_index], output], axis=1)
                pred = tf.add(tf.matmul(H_and_output, W_softmax), b_softmax)
                predictions = tf.nn.softmax(pred)
                predicted_indices = tf.argmax(predictions, axis=1)
                outputs.append(predicted_indices)

            self.outputs = tf.stack(outputs, axis=1)

        return self.outputs

    def beam_decoding(self, H, config):

        #batch size
        b_size = tf.shape(H)[0]

        """Reload softmax prediction layer"""
        with tf.variable_scope("softmax", reuse=True):
            W_softmax = tf.get_variable("W_softmax")
            b_softmax = tf.get_variable("b_softmax")

        #reloading the tag embedding layer.
        with tf.variable_scope("tag_embedding_layer", reuse=True):
            tag_lookup_table = tf.get_variable("tag_lookup_table")

        GO_symbol = tf.zeros((b_size, config.tag_embedding_size), dtype=tf.float32)
        GO_context = tf.zeros((b_size, config.word_rnn_hidden_units), dtype=tf.float32)
        initial_state = self.decoder_lstm_cell.zero_state(b_size, tf.float32)
        H_t = tf.transpose(H, [1,0,2])
        outputs = []

        """ we will need index to select top ranked beamsize stuff"""
        #batch index
        b_index = tf.reshape(tf.range(0, b_size),(b_size, 1))

        #beam index
        be_index = tf.constant(
                                config.beamsize * config.beamsize,
                                dtype=tf.int32,
                                shape=(1, config.beamsize)
                                )


        with tf.variable_scope("decoder_rnn", reuse=True) as scope:
            for time_index in range(config.max_sentence_length):
                if time_index==0:
                    inp = tf.concat([GO_symbol, GO_context], axis=1)
                    output, (c_state, m_state) = self.decoder_lstm_cell(inp, initial_state)

                    H_and_output = tf.concat([H_t[time_index], output], axis=1)
                    pred = tf.add(tf.matmul(H_and_output, W_softmax), b_softmax)
                    predictions = tf.nn.softmax(pred)
                    probs, indices = tf.nn.top_k(predictions, k=config.beamsize, sorted=True)
                    prev_indices = indices
                    beam = tf.expand_dims(indices, axis=2)
                    prev_probs = tf.log(probs)
                    prev_c_states = [c_state for i in range(config.beamsize)]
                    prev_c_states = tf.stack(prev_c_states, axis=1)
                    prev_m_states = [m_state for i in range(config.beamsize)]
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
                    for b in range(config.beamsize):
                        prev_output = tf.nn.embedding_lookup(tag_lookup_table, prev_indices_t[b])
                        inp = tf.concat([prev_output, H_t[time_index-1]], axis=1)
                        output, (c_state, m_state) = self.decoder_lstm_cell(
                                                        inp,
                                                        (prev_c_states_t[b],prev_m_states_t[b])
                                                        )
                        H_and_output = tf.concat([H_t[time_index], output], axis=1)
                        pred = tf.add(tf.matmul(H_and_output, W_softmax), b_softmax)

                        predictions = tf.nn.softmax(pred)
                        probs, indices = tf.nn.top_k(predictions, k=config.beamsize, sorted=True)
                        probs_t = tf.transpose(probs, [1,0])
                        indices_t = tf.transpose(indices, [1,0])
                        for bb in range(config.beamsize):
                            probs_candidates.append(tf.add(prev_probs_t[b], tf.log(probs_t[bb])))
                            indices_candidates.append(indices_t[bb])
                            beam_candidates.append(tf.concat(
                                                        [beam_t[b],
                                                         tf.expand_dims(indices_t[bb], axis=1)
                                                         ], axis=1
                                                         )
                                                    )
                            c_state_candidates.append(c_state)
                            m_state_candidates.append(m_state)

                    temp_probs = tf.stack(probs_candidates, axis=1)
                    temp_indices = tf.stack(indices_candidates, axis=1)
                    temp_beam = tf.stack(beam_candidates, axis=1)
                    temp_c_states = tf.stack(c_state_candidates, axis=1)
                    temp_m_states = tf.stack(m_state_candidates, axis=1)
                    _, max_indices = tf.nn.top_k(temp_probs, k=config.beamsize, sorted=True)

                    #index
                    index = tf.add(
                                tf.matmul(b_index, be_index),
                                max_indices
                                )
                    prev_probs = tf.gather(tf.reshape(temp_probs, [-1]), index)
                    prev_indices = tf.gather(tf.reshape(temp_indices, [-1]), index)
                    beam = tf.gather(tf.reshape(temp_beam, [-1, time_index+1]), index)
                    prev_c_states = tf.gather(
                                            tf.reshape(
                                                temp_c_states,
                                                [-1, config.decoder_rnn_hidden_units]
                                            ),
                                            index
                                        )
                    prev_m_states = tf.gather(
                                            tf.reshape(
                                                temp_m_states,
                                                [-1, config.decoder_rnn_hidden_units]
                                            ),
                                            index
                                        )

            beam_t = tf.transpose(beam, [1,0,2])
            outputs = beam_t[0]

        return outputs

    def add_training_op(self, loss, config):
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
        with tf.variable_scope("adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, config.max_gradient_norm)
        train_operation = optimizer.apply_gradients(zip(clipped_gradients, variables))

        return train_operation
