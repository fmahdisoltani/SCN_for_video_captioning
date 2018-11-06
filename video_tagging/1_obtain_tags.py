import numpy as np
import cPickle


#### CURRENT EPIC FEATURES ####
#CORPUS_P_PATH = '../../../data/epic/corpus_epic.p'
#REFERENCES_PATH = '../../../data/epic/references_epic.p'
#GT_TAG_FEATS_PATH = '../../../data/epic/gt_tag_feats_epic.p'
#NUM_TAGS = 1195

#### CURRENT BREAKFAST FEATURES ####
CORPUS_P_PATH = '../../../data/breakfast_current/corpus_breakfast_current.p'
REFERENCES_PATH = '../../../data/breakfast_current/references_breakfast_current.p'
GT_TAG_FEATS_PATH = '../../../data/breakfast_current/gt_tag_feats_breakfast_current.p'
NUM_TAGS = 48

if __name__ == "__main__":
    
    x = cPickle.load(open(CORPUS_P_PATH,"rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    del x
    n_words = len(ixtoword)
    
    n_count = np.zeros((n_words,)).astype("int32")
    for sent in train[0]:
        for w in sent:
            n_count[w] = n_count[w] + 1
    
    for sent in val[0]:
        for w in sent:
            n_count[w] = n_count[w] + 1
    
    idx = np.argsort(n_count)[::-1]
    
    count_sorted = np.sort(n_count)[::-1]
    word_sorted = []
    for i in idx:
        word_sorted.append(ixtoword[i])
    # manually select tags that you think are important and useful 
    # here, we manually select 1195 (377 for sample epic) tags when we did the experiments
    num_tags = NUM_TAGS
    selected = range(1,NUM_TAGS + 1)
                
    #print len(selected)
    #print selected
    key_words = []
    for i in selected:
        key_words.append(word_sorted[i])
    
    ixtoword = {}
    wordtoix = {}
    
    for idx in range(len(key_words)):
        wordtoix[key_words[idx]] = idx
        ixtoword[idx] = key_words[idx]
        
    x = cPickle.load(open(REFERENCES_PATH,"rb"))
    train_refs, valid_refs, test_refs = x[0], x[1], x[1]
    del x
    
    train_label = np.zeros((len(train_refs),num_tags))
    for i in range(len(train_refs)):
        sents = train_refs[i]
        for sent in sents:
            words = sent.split(" ")
            for w in words:
                if w in wordtoix:
                    train_label[i,wordtoix[w]] = 1.
                    
    valid_label = np.zeros((len(valid_refs),num_tags))
    for i in range(len(valid_refs)):
        sents = valid_refs[i]
        for sent in sents:
            words = sent.split(" ")
            for w in words:
                if w in wordtoix:
                    valid_label[i,wordtoix[w]] = 1.
                    
    test_label = np.zeros((len(test_refs),num_tags))
    for i in range(len(test_refs)):
        sents = test_refs[i]
        for sent in sents:
            words = sent.split(" ")
            for w in words:
                if w in wordtoix:
                    test_label[i,wordtoix[w]] = 1.
    cPickle.dump([train_label, valid_label, test_label, wordtoix, ixtoword], open(GT_TAG_FEATS_PATH, "wb"))
    1 / 0        
