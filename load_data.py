import numpy as np
import utils as ut
import re

def load_data(config):
    """
    Loads starter word vectors, creates random character vectors,
    and loads train/dev/test data.
    """
    #Loads the starter word vectors
    print "INFO: Reading word embeddings!"
    word_vectors, words = ut.load_embeddings(
                                      config.word_dic_path,
                                      config.word_vectors_path
                                      )

    num_to_word = dict(enumerate(words))
    word_to_num = {v:k for k,v in num_to_word.iteritems()}

    # Load the starter char vectors
    print "INFO: Reading character embeddings!"
    _, chars = ut.load_embeddings(
                      config.char_dic_path,
                      None
                      )

    num_to_char = dict(enumerate(chars))
    char_to_num = {v:k for k,v in num_to_char.iteritems()}
    config.char_size = len(chars)

    # For IOBES format
    tagnames = ['O', 'B-LOC', 'I-LOC', 'S-LOC', 'E-LOC',
                  'B-ORG', 'I-ORG', 'S-ORG', 'E-ORG',
                  'B-PER','I-PER', 'S-PER', 'E-PER',
                  'B-MISC', 'I-MISC', 'S-MISC', 'E-MISC']

    num_to_tag = dict(enumerate(tagnames))
    tag_to_num = {v:k for k,v in num_to_tag.iteritems()}

    #Loads the training set
    print "INFO: Reading training data!"
    docs = ut.load_dataset(config.train_set_path)
    Char_X_train, Cap_X_train, Word_X_train, Y_train = ut.docs_to_sentences(
                                                              docs,
                                                              word_to_num,
                                                              tag_to_num,
                                                              char_to_num
                                                              )
    train_data = ut.padding(
                      Char_X_train,
                      Cap_X_train,
                      Word_X_train,
                      Y_train,
                      config.max_sentence_length,
                      config.max_word_length
                      )

    #Loads the dev set (for tuning hyperparameters)
    print "INFO: Reading dev data!"
    docs = ut.load_dataset(config.dev_set_path)
    Char_X_dev, Cap_X_dev, Word_X_dev, Y_dev = ut.docs_to_sentences(
                                                      docs,
                                                      word_to_num,
                                                      tag_to_num,
                                                      char_to_num
                                                      )

    dev_data = ut.padding(
                      Char_X_dev,
                      Cap_X_dev,
                      Word_X_dev,
                      Y_dev,
                      config.max_sentence_length,
                      config.max_word_length
                      )


    # Loads the test set
    print "INFO: Reading test data!"
    docs = ut.load_dataset(config.test_set_path)
    Char_X_test, Cap_X_test, Word_X_test, Y_test = ut.docs_to_sentences(
                                                          docs,
                                                          word_to_num,
                                                          tag_to_num,
                                                          char_to_num
                                                          )
    test_data = ut.padding(
                      Char_X_test,
                      Cap_X_test,
                      Word_X_test,
                      Y_test,
                      config.max_sentence_length,
                      config.max_word_length
                      )

    ret = {}
    ret['num_to_tag'] = num_to_tag
    ret['num_to_word'] = num_to_word
    ret['word_vectors'] = word_vectors
    ret['train_data'] = train_data
    ret['dev_data'] = dev_data
    ret['test_data'] = test_data

    return ret
