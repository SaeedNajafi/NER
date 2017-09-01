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
    early_stopping = 2
    random_seed = 11

    """inference type"""
    inference = "softmax"
    #inference = "crf"
    #inference = "decoder_rnn"
    #inference = "crf_rnn"

    """for decoder_rnn and crf_rnn"""
    #decoding="greedy"
    #decoding="beamsearch"
    #beamsize=4
    #crf_beamsize=tag_size

    """path to different files"""
    word_dic_path = './data/glove_en_word_100_dic.txt'
    word_vectors_path = './data/glove_en_word_100_vectors.txt'
    char_dic_path = './data/en_char_dic.txt'
    train_set_path = './data/eng.train.v1'
    dev_set_path = './data/eng.testa.v1'
    test_set_path = './data/eng.testb.v1'
