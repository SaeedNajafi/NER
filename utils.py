import re
import itertools
from numpy import *
import numpy as np

def load_embeddings(vocabfile, vectorfile=None):
    em = None
    if(vectorfile is not None):
        em = np.loadtxt(vectorfile, dtype=np.float32)
    with open(vocabfile) as fd:
        tokens = [line.strip() for line in fd]
    return em, tokens

def load_dataset(fname):
    docs = []
    with open(fname) as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else: # read in tokens
                cur.append(line.strip().split("\t",1))

        # flush running buffer
        docs.append(cur)
    return docs

def docs_to_sentences(docs, word_to_num, tag_to_num, char_to_num):
    pad = 1
    docs = flatten([pad_sequence(seq, left=pad, right=pad) for seq in docs])
    words, tags = zip(*docs)
    unchanged_words = [canonicalize_char(w, char_to_num) for w in words]
    caps = [capalize_word(w) for w in words]
    words = [canonicalize_word(w, word_to_num) for w in words]
    tags = [t.split("|")[0] for t in tags]

    #converting tags from IOB format to IOB2.
    iob2(tags)

    #converting tags from IOB2 format to IOBES format.
    tags = iob_iobes(tags)

    return seq_to_sentences(
                        unchanged_words,
                        caps,
                        words,
                        tags,
                        word_to_num,
                        tag_to_num,
                        char_to_num
                        )

#https://github.com/glample/tagger/blob/master/utils.py
def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

#https://github.com/glample/tagger/blob/master/utils.py
def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

#https://github.com/glample/tagger/blob/master/utils.py
def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True

#https://github.com/glample/tagger/blob/master/loader.py
def capalize_word(word):
    """
        Capitalization feature:
        0 = low caps without digits
        1 = all caps without digits
        2 = first letter caps without digits
        3 = one capital (not first letter) without digits
    """
    if word.lower() == word:
	return 0

    elif word.upper() == word:
        return 1

    elif word[0].upper() == word[0]:
    	return 2

    else:
    	return 3

def canonicalize_word(word, wordset):
	if word in wordset: return word
	elif word.lower() in wordset: return word.lower()
	elif re.search(r'\d', word):
		return "number" # for glove
	else:
		return "unknown" # for glove

def canonicalize_char(word, charset):
	word = word.lower()
	new_word = ""
	for each in list(word):
		if each in charset:
			new_word += each
	return new_word

def seq_to_sentences(
        unchanged_words,
        caps,
        words,
        tags,
        word_to_num,
        tag_to_num,
        char_to_num):

    ns = len(words)

    Cap_X = []
    Word_X = []
    Y = []
    Char_X = []

    cap_x = []
    word_x = []
    y = []
    char_x = []

    for i in range(ns):
        tag = tags[i]
        word = words[i]
        cap = caps[i]
        unchanged_word = unchanged_words[i]

        if word != "<s>" and word != "</s>":
            y.append(tag_to_num[tag])
            word_x.append(word_to_num[word])
            cap_x.append(cap)
            char_x.append([char_to_num[chr] for chr in list(unchanged_word)])

        elif word == "<s>":
            cap_x = []
            word_x = []
            y = []
            char_x = []

        else:
            Cap_X.append(cap_x)
            Word_X.append(word_x)
            Y.append(y)
            Char_X.append(char_x)

    return array(Char_X), array(Cap_X), array(Word_X), array(Y)

def pad_sequence(seq, left=1, right=1):
    return left*[("<s>", "O")] + seq + right*[("</s>", "O")]

def flatten(lst):
    return list(itertools.chain.from_iterable(lst))

def padding(
        char_data,
        cap_data,
        word_data,
        tag_data,
        max_sentence_length,
        max_word_length):

    cap_X = []
    word_X = []
    mask_X = []
    sentence_length_X = []
    Y = []

    char_X = []
    word_length_X = []

    for index in range(len(word_data)):
        sentence = word_data[index]
        tags = tag_data[index]
        caps = cap_data[index]
        sentence_length_X.append(len(sentence))
        j = len(sentence)
        mask_list = [1.0] * j
        if j < max_sentence_length:
            while j < max_sentence_length:
                sentence.append(0)
                tags.append(0)
                caps.append(0)
                mask_list.append(0.0)
                j += 1
        else:
            sentence = sentence[0:max_sentence_length]
            tags = tags[0:max_sentence_length]
            caps = caps[0:max_sentence_length]
            mask_list = mask_list[0:max_sentence_length]

        cap_X.append(caps)
        word_X.append(sentence)
        mask_X.append(mask_list)
        Y.append(tags)

    for index in range(len(char_data)):
        sentence = char_data[index]
        pad_list = [0] * max_word_length
        length = []
        new_sentence = []
        for k in range(len(sentence)):
            word = sentence[k]
            length.append(len(word))
            j = len(word)
            if j < max_word_length:
                while j < max_word_length:
                    word.append(0)
                    j += 1
            else:
                word = word[0:max_word_length]


            new_sentence.append(word)

        j = len(new_sentence)
        if j < max_sentence_length:
            while j < max_sentence_length:
                new_sentence.append(pad_list)
                length.append(0)
                j += 1
        else:
            new_sentence = new_sentence[0:max_sentence_length]
            length = length[0:max_sentence_length]

        word_length_X.append(length)
        char_X.append(new_sentence)

    ret = {
            'char_X': array(char_X),
            'word_length_X': array(word_length_X),
            'cap_X': array(cap_X),
            'word_X': array(word_X),
            'mask_X': array(mask_X),
            'sentence_length_X': array(sentence_length_X),
            'Y': array(Y)
            }

    return ret

def data_iterator(
        orig_char_X,
        orig_word_length_X,
        orig_cap_X,
        orig_word_X,
        orig_mask_X,
        orig_sentence_length_X,
        batch_size,
        tag_size,
        orig_Y,
        shuffle
        ):

    data_cap_X, data_char_X, data_word_length_X = orig_cap_X, orig_char_X, orig_word_length_X
    data_word_X, data_mask_X  = orig_word_X, orig_mask_X
    data_sentence_length_X, data_Y = orig_sentence_length_X, orig_Y

    total_steps = int(np.ceil(len(data_word_X) / float(batch_size)))
    if shuffle:
    	steps = np.random.permutation(total_steps).tolist()
    else:
    	steps = range(total_steps)

    for step in steps:
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        ret_char_X = data_char_X[batch_start:batch_start + batch_size][:][:]
        ret_word_length_X = data_word_length_X[batch_start:batch_start + batch_size][:]
        ret_word_X = data_word_X[batch_start:batch_start + batch_size][:]
        ret_mask_X = data_mask_X[batch_start:batch_start + batch_size][:]
        ret_sentence_length_X = data_sentence_length_X[batch_start:batch_start + batch_size]
        ret_cap_X = data_cap_X[batch_start:batch_start + batch_size][:]

        ret_Y = None
        if np.any(data_Y):
            ret_Y = data_Y[batch_start:batch_start + batch_size][:]

        ###
        yield ret_char_X, ret_word_length_X, ret_cap_X, ret_word_X, ret_mask_X, ret_sentence_length_X, ret_Y
