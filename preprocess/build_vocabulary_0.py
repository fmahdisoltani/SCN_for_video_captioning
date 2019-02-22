##This file builds vocabulary.txt

import numpy as np
import cPickle
import scipy.io

#### CURRENT EPIC FEATURES ####
#CAPTION_TRAIN_PATH = '../../../data/epic_current/current_epic_video_description_train.txt'
#CAPTION_VALID_PATH = '../../../data/epic_current/current_epic_video_description_val.txt'
#VOCAB_PATH = '../../../data/epic_current/vocabulary_epic_current.txt'

#### FUTURE EPIC FEATURES ####
#CAPTION_TRAIN_PATH = '../../../data/epic_futureRC/future_video_description_train.txt'
#CAPTION_VALID_PATH = '../../../data/epic_futureRC/future_video_description_val.txt'
#VOCAB_PATH = '../../../data/epic_futureRC/vocabulary_epic_future.txt'

#### CF EPIC FEATURES ####
#CAPTION_TRAIN_PATH = '../../../data/epic_CF/CF_future_video_description_train.txt'
#CAPTION_VALID_PATH = '../../../data/epic_CF/CF_future_video_description_val.txt'
#VOCAB_PATH = '../../../data/epic_CF/vocabulary_epic_CF.txt'

#### CURRENT BREAKFAST FEATURES ####
#CAPTION_TRAIN_PATH = '../../data/breakfast_current/current_s1_train_caption_list'
#CAPTION_VALID_PATH = '../../data/breakfast_current/current_s1_val_caption_list'
#VOCAB_PATH = '../../data/breakfast_current/vocabulary_breakfast_current.txt'

#### FUTURE BREAKFAST FEATURES ####
#CAPTION_TRAIN_PATH = '../../data/breakfast_future/future_s1_train_caption_list'
#CAPTION_VALID_PATH = '../../data/breakfast_future/future_s1_val_caption_list'
#VOCAB_PATH = '../../data/breakfast_future/vocabulary_breakfast_future.txt'

#### FUTURE RC BREAKFAST FEATURES ####
#CAPTION_TRAIN_PATH = '../../data/breakfast_futureRC/futureRC_s1_train_caption_list'
#CAPTION_VALID_PATH = '../../data/breakfast_futureRC/futureRC_s1_val_caption_list'
#VOCAB_PATH = '../../data/breakfast_futureRC/vocabulary_breakfast_futureRC.txt'

#### CF BREAKFAST FEATURES ####
#CAPTION_TRAIN_PATH = '../../data/breakfast_CF/CF_s1_train_caption_list'
#CAPTION_VALID_PATH = '../../data/breakfast_CF/CF_s1_val_caption_list'
#VOCAB_PATH = '../../data/breakfast_CF/vocabulary_breakfast_CF.txt'

#### CF youcook FEATURES ####
#CAPTION_TRAIN_PATH = '../../../data/youcook2_CF/CF_youcook_description_training_future'
#CAPTION_VALID_PATH = '../../../data/youcook2_CF/CF_youcook_description_validation_future'
#VOCAB_PATH = '../../../data/youcook2_CF/vocabulary_youcook2_CF.txt'

#### future youcook FEATURES ####
#CAPTION_TRAIN_PATH = '../../data/youcook2_future/future_youcook_description_training_future'
#CAPTION_VALID_PATH = '../../data/youcook2_future/future_youcook_description_validation_future'
#VOCAB_PATH = '../../data/youcook2_future/vocabulary_youcook2_future.txt'

#if __name__ == "__main__":
def step0_build_vocab(config_obj):
    CAPTION_TRAIN_PATH = config_obj.get('paths', 'caption_train_path')
    CAPTION_VALID_PATH = config_obj.get('paths', 'caption_valid_path')
    VOCAB_PATH = config_obj.get('paths', 'vocab_path')
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
    num_tags = 0
    with open(VOCAB_PATH, 'w') as f:
        for item in vocab:
             print >> f, item
             num_tags  +=1
    print "0..Vocabulary Built"
    return num_tags


