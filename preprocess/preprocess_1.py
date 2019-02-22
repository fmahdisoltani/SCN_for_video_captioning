##This file builds references.p and data.p

import numpy as np
import cPickle
import scipy.io

#### CURRENT EPIC FEATURES ####
#VOCAB_PATH = '../../../data/epic_current/vocabulary_epic_current.txt'
#TRAIN_CAPTION_PATH = '../../../data/epic_current/current_epic_video_description_train.txt'
#VALID_CAPTION_PATH = '../../../data/epic_current/current_epic_video_description_val.txt'
#DATA_P_PATH = '../../../data/epic_current/data_epic_current.p'
#REFERENCES_PATH = '../../../data/epic_current/references_epic_current.p'

#### future RC EPIC FEATURES ####
#VOCAB_PATH = '../../../data/epic_futureRC/vocabulary_epic_future.txt'
#TRAIN_CAPTION_PATH = '../../../data/epic_futureRC/future_video_description_train.txt'
#VALID_CAPTION_PATH = '../../../data/epic_futureRC/future_video_description_val.txt'
#DATA_P_PATH = '../../../data/epic_futureRC/data_epic_future.p'
#REFERENCES_PATH = '../../../data/epic_futureRC/references_epic_future.p'
#
#### CF EPIC FEATURES ####
#VOCAB_PATH = '../../../data/epic_CF/vocabulary_epic_CF.txt'
#TRAIN_CAPTION_PATH =  '../../../data/epic_CF/CF_future_video_description_train.txt'
#VALID_CAPTION_PATH =  '../../../data/epic_CF/CF_future_video_description_val.txt'
#DATA_P_PATH = '../../../data/epic_CF/data_epic_CF.p'
#REFERENCES_PATH = '../../../data/epic_CF/references_epic_CF.p'

#### CURRENT BREAKFAST FEATURES ####
#VOCAB_PATH = '../../data/breakfast_current/vocabulary_breakfast_current.txt'
#TRAIN_CAPTION_PATH =  '../../data/breakfast_current/current_s1_train_caption_list'
#VALID_CAPTION_PATH =  '../../data/breakfast_current/current_s1_val_caption_list'
#DATA_P_PATH = '../../data/breakfast_current/data_breakfast_current.p'
#REFERENCES_PATH = '../../data/breakfast_current/references_breakfast_current.p'

##### FUTURE BREAKFAST FEATURES ####
#VOCAB_PATH = '../../data/breakfast_future/vocabulary_breakfast_future.txt'
#TRAIN_CAPTION_PATH =  '../../data/breakfast_future/future_s1_train_caption_list'
#VALID_CAPTION_PATH =  '../../data/breakfast_future/future_s1_val_caption_list'
#DATA_P_PATH = '../../data/breakfast_future/data_breakfast_future.p'
#REFERENCES_PATH = '../../data/breakfast_future/references_breakfast_future.p'

#### CF BREAKFAST FEATURES ####
#VOCAB_PATH = '../../data/breakfast_CF/vocabulary_breakfast_CF.txt'
#TRAIN_CAPTION_PATH =  '../../data/breakfast_CF/CF_s1_train_caption_list'
#VALID_CAPTION_PATH =  '../../data/breakfast_CF/CF_s1_val_caption_list'
#DATA_P_PATH = '../../data/breakfast_CF/data_breakfast_CF.p'
#REFERENCES_PATH = '../../data/breakfast_CF/references_breakfast_CF.p'

##### FUTURE RC BREAKFAST FEATURES ####
#VOCAB_PATH = '../../data/breakfast_futureRC/vocabulary_breakfast_futureRC.txt'
#TRAIN_CAPTION_PATH =  '../../data/breakfast_futureRC/futureRC_s1_train_caption_list'
#VALID_CAPTION_PATH =  '../../data/breakfast_futureRC/futureRC_s1_val_caption_list'
#DATA_P_PATH = '../../data/breakfast_futureRC/data_breakfast_futureRC.p'
#REFERENCES_PATH = '../../data/breakfast_futureRC/references_breakfast_futureRC.p'

#### CF EPIC FEATURES ####
#VOCAB_PATH = '../../../data/epic_CF/vocabulary_epic_CF.txt'
#TRAIN_CAPTION_PATH =  '../../../data/epic_CF/CF_future_video_description_train.txt'
#VALID_CAPTION_PATH =  '../../../data/epic_CF/CF_future_video_description_val.txt'
#DATA_P_PATH = '../../../data/epic_CF/data_epic_CF.p'
#REFERENCES_PATH = '../../../data/epic_CF/references_epic_CF.p'


#### CF youcook2 FEATURES ####
#VOCAB_PATH = '../../../data/youcook2_CF/vocabulary_youcook2_CF.txt'
#TRAIN_CAPTION_PATH = '../../../data/youcook2_CF/CF_youcook_description_training_future'
#VALID_CAPTION_PATH = '../../../data/youcook2_CF/CF_youcook_description_validation_future'
#DATA_P_PATH = '../../../data/youcook2_CF/data_youcook2_CF.p'
#REFERENCES_PATH = '../../../data/youcook2_CF/references_youcook2_CF.p'

#### future youcook2 FEATURES ####
#VOCAB_PATH = '../../data/youcook2_future/vocabulary_youcook2_future.txt'
#TRAIN_CAPTION_PATH = '../../data/youcook2_future/future_youcook_description_training_future'
#VALID_CAPTION_PATH = '../../data/youcook2_future/future_youcook_description_validation_future'
#DATA_P_PATH = '../../data/youcook2_future/data_youcook2_future.p'
#REFERENCES_PATH = '../../data/youcook2_future/references_youcook2_future.p'


def step1_preprocess(config_obj):
#if __name__ == "__main__":
    CAPTION_TRAIN_PATH = config_obj.get('paths', 'caption_train_path')
    CAPTION_VALID_PATH = config_obj.get('paths', 'caption_valid_path')
    VOCAB_PATH = config_obj.get('paths', 'vocab_path')

    DATA_P_PATH = config_obj.get('paths', 'data_p_path')
    REFERENCES_PATH =config_obj.get('paths', 'references_path')



    ixtoword = {}
    ixtoword[0] = '<eos>'
    wordtoix = {}
    wordtoix['<eos>'] = 0
     
    ix = 1
    with open(VOCAB_PATH, 'rb') as f:
        for line in f:
            word = line.strip()
            wordtoix[word] = ix
            ixtoword[ix] = word
            ix = ix + 1
     
    train_id = []
    train_ref = []
    train_cap = []     
    with open(CAPTION_TRAIN_PATH, 'rb') as f:
        for count, line in enumerate(f):
            #print(count)
            #print(line)
            #tmp1 = line.strip().split("\t")
            tmp1 = line.strip().replace(".","")
            tmp2 = tmp1.split(" ")
            tmp3 = []
            for w in tmp2:
                if w in wordtoix:
                    tmp3.append(wordtoix[w])
                else:
                    print(w)
                    print('not found')
                    tmp3.append(1)
            tmp3.append(0)
            train_cap.append(tmp3)
            train_ref.append(tmp2)
             
            #tmp4 = tmp1[0].split("d")
            #tmp5 = int(tmp4[1])-1
            train_id.append(count)
    
    train_count = count
    valid_id = []
    valid_ref =[]
    valid_cap = []
    with open(CAPTION_VALID_PATH, 'rb') as f:
        for count, line in enumerate(f):
            #print(count)
            #print(line)
            #tmp1 = line.strip().split("\t")
            tmp1 = line.strip().replace(".","")
            tmp2 = tmp1.split(" ")
            tmp3 = []
            for w in tmp2:
                if w in wordtoix:
                    tmp3.append(wordtoix[w])
                else:
                    print (w)
                    print('not found')
                    tmp3.append(1)
            tmp3.append(0)
            valid_cap.append(tmp3)
            valid_ref.append(tmp2)
             
            #tmp4 = tmp1[0].split("d")
            #tmp5 = int(tmp4[1])-1
            valid_id.append(count)
    test_id = valid_id
    test_ref = valid_ref 
    test_cap = valid_cap
     
    print "show one example:"
    tt = []
    for i in test_cap[34]:
        tt.append(ixtoword[i])
    print " ".join(tt)
     
    train = [train_cap,train_id]
    valid = [valid_cap,valid_id]
    test = [test_cap, test_id]
    cPickle.dump([train, valid, test,wordtoix, ixtoword], open(DATA_P_PATH, "wb"))


    """ generate a file so that we can calculate BLEU scores on test. """     
    train_id = []
    train_cap = []
    with open(CAPTION_TRAIN_PATH, 'rb') as f:
        for count, line in enumerate(f):
            tmp1 = line.strip().replace(".","").split(" ")
            train_cap.append(tmp1)
             
            #tmp2 = tmp1[0].split("d")
            train_id.append(count)
    train_count = count

    valid_id = []
    valid_cap = []
    with open(CAPTION_VALID_PATH, 'rb') as f:
        for count, line in enumerate(f):
            tmp1 = line.strip().replace("."," ").split(" ")
            valid_cap.append(tmp1)
             
            #tmp2 = tmp1[0].split("d")
            valid_id.append(count+train_count+1)
    ################################################################     
    cPickle.dump([train_ref, valid_ref, test_ref], open(REFERENCES_PATH, "wb"))
print "1..Preprocessing Done"
