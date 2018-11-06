##This file builds references.p and data.p

import numpy as np
import cPickle
import scipy.io

#### CURRENT EPIC FEATURES ####
#VOCAB_PATH = '../../../data/epic/vocabulary_epic.txt'
#TRAIN_CAPTION_PATH = '../../../data/epic/video_description_train.txt'
#VALID_CAPTION_PATH = '../../../data/epic/video_description_val.txt'
#DATA_P_PATH = '../../../data/epic/data_epic.p'
#REFERENCES_PATH = '../../../data/epic/references_epic.p'

#### CURRENT BREAKFAST FEATURES ####
VOCAB_PATH = '../../../data/breakfast_current/vocabulary_breakfast_current.txt'
TRAIN_CAPTION_PATH =  '../../../data/breakfast_current/current_s1_train_caption_list'
VALID_CAPTION_PATH =  '../../../data/breakfast_current/current_s1_val_caption_list'
DATA_P_PATH = '../../../data/breakfast_current/data_breakfast_current.p'
REFERENCES_PATH = '../../../data/breakfast_current/references_breakfast_current.p'


if __name__ == "__main__":
     
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
    with open(TRAIN_CAPTION_PATH, 'rb') as f:
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
    with open(VALID_CAPTION_PATH, 'rb') as f:
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
    with open(TRAIN_CAPTION_PATH, 'rb') as f:
        for count, line in enumerate(f):
            tmp1 = line.strip().replace(".","").split(" ")
            train_cap.append(tmp1)
             
            #tmp2 = tmp1[0].split("d")
            train_id.append(count)
    train_count = count

    valid_id = []
    valid_cap = []
    with open(VALID_CAPTION_PATH, 'rb') as f:
        for count, line in enumerate(f):
            tmp1 = line.strip().replace("."," ").split(" ")
            valid_cap.append(tmp1)
             
            #tmp2 = tmp1[0].split("d")
            valid_id.append(count+train_count+1)
    ################################################################     
    cPickle.dump([train_ref, valid_ref, test_ref], open(REFERENCES_PATH, "wb"))
