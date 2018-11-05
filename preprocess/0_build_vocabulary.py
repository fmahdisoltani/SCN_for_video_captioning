##This file builds vocabulary.txt

import numpy as np
import cPickle
import scipy.io

if __name__ == "__main__":

    ixtoword = {}
    ixtoword[0] = '<eos>'
    wordtoix = {}
    wordtoix['<eos>'] = 0

    vocab = set()
    with open('../../../data/epic/video_description_train.txt', 'rb') as f:
        for line in f:
            words = line.strip().replace(".","").split(" ")
            for w in words:
                vocab.add(w)
    # with open('../data/epic/vocabulary_epic.txt', 'w') as f:
    #    for item in vocab:
    #         print >> f, item


    with open('../../../data/epic/video_description_val.txt', 'rb') as f:
        for line in f:
            words = line.strip().replace(".","").split(" ")
            for w in words:
                vocab.add(w)
    with open('../../../data/epic/vocabulary_epic.txt', 'w') as f:
        for item in vocab:
             print >> f, item
