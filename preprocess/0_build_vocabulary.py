##This file builds vocabulary.txt

import numpy as np
import cPickle
import scipy.io

#### CURRENT EPIC FEATURES ####
#CAPTION_TRAIN_PATH = '../../../data/epic/video_description_train.txt'
#CAPTION_VALID_PATH = '../../../data/epic/video_description_val.txt'
#VOCAB_PATH = '../../../data/epic/vocabulary_epic.txt'

#### CURRENT BREAKFAST FEATURES ####
CAPTION_TRAIN_PATH = '../../../data/breakfast_current/current_s1_train_caption_list'
CAPTION_VALID_PATH = '../../../data/breakfast_current/current_s1_val_caption_list'
VOCAB_PATH = '../../../data/breakfast_current/vocabulary_breakfast_current.txt'

if __name__ == "__main__":

    ixtoword = {}
    ixtoword[0] = '<eos>'
    wordtoix = {}
    wordtoix['<eos>'] = 0

    vocab = set()
    with open(CAPTION_TRAIN_PATH, 'rb') as f:
        for line in f:
            words = line.strip().replace(".","").split(" ")
            for w in words:
                vocab.add(w)
    with open(VOCAB_PATH, 'w') as f:
       for item in vocab:
           print >> f, item


    with open(CAPTION_VALID_PATH, 'rb') as f:
        for line in f:
            words = line.strip().replace(".","").split(" ")
            for w in words:
                vocab.add(w)
    with open(VOCAB_PATH, 'w') as f:
        for item in vocab:
             print >> f, item
