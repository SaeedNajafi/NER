class Configuration(object):

    """path to different files"""
    word_dic_path = './en_data/glove_en_word_100_dic.txt'
    word_vectors_path = './en_data/glove_en_word_100_vectors.txt'
    char_dic_path = './en_data/en_char_dic.txt'
    train_set_path = './en_data/eng.train.v1.txt'
    dev_set_path = './en_data/eng.testa.v1.txt'
    test_set_path = './en_data/eng.testb.v1.txt'

    """Model hyperparams and data information"""
    word_embedding_size = 100
    word_rnn_hidden_units = 128
    char_embedding_size = 32
    char_rnn_hidden_units = 32
    max_sentence_length = 150
    max_word_length = 32
    tag_size = 17
    decoder_rnn_hidden_units = 128
    tag_embedding_size = 32

    #to speedup experiments
    batch_size = 128

    dropout = 0.5
    learning_rate = 0.0005
    max_gradient_norm = 5.
    max_epochs = 64
    early_stopping = 5
    runs=10
    gamma = 0.7
    n_step = 3

    #inference = "CRF"
    #inference = "RNN"
    #inference = "AC-RNN"
    #inference = "INDP"
    #inference = "DIF-SCH"
    #inference = "SCH"
    #inference = "R-RNN"
    #inference = "BR-RNN"

    beamsize = 4
