class Configuration(object):
    """Model hyperparams and data information"""

    word_embedding_size = 100
    word_rnn_hidden_units = 200

    char_embedding_size = 25
    char_rnn_hidden_units = 25

    max_sentence_length = 150
    max_word_length = 25

    tag_size = 17
    decoder_rnn_hidden_units = 34
    tag_embedding_size = 17

    batch_size = 10
    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 48
    early_stopping = 3
    random_seed = 11
    crf_beamsize = 8
    """inference type"""
    #inference = "softmax"
    #inference = "crf"
    #inference = "decoder_rnn"
    inference = "scheduled_decoder_rnn"

    """for decoder_rnn"""
    decoding="greedy"
    #decoding="beamsearch"
    #beamsize=4

    """path to different files"""
    word_dic_path = './en_data/glove_en_word_100_dic.txt'
    word_vectors_path = './en_data/glove_en_word_100_vectors.txt'
    char_dic_path = './en_data/en_char_dic.txt'
    train_set_path = './en_data/eng.train.v1'
    dev_set_path = './en_data/eng.testa.v1'
    test_set_path = './en_data/eng.testb.v1'
