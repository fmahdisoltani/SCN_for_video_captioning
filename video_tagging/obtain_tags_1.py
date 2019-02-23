import numpy as np
import cPickle


#### CURRENT EPIC FEATURES ####
#CORPUS_P_PATH = '../../../data/epic_current/corpus_epic_current.p'
#REFERENCES_PATH = '../../../data/epic_current/references_epic_current.p'
#GT_TAG_FEATS_PATH = '../../../data/epic_current/gt_tag_feats_epic_current.p'
#NUM_TAGS = 1192
#
##### FUTURE EPIC FEATURES ####
#CORPUS_P_PATH = '../../../data/epic_futureRC/corpus_epic_future.p'
#REFERENCES_PATH = '../../../data/epic_futureRC/references_epic_future.p'
#GT_TAG_FEATS_PATH = '../../../data/epic_futureRC/gt_tag_feats_epic_future.p'
#NUM_TAGS = 1192
#
#### CF EPIC FEATURES ####
#CORPUS_P_PATH = '../../../data/epic_CF/corpus_epic_CF.p'
#REFERENCES_PATH = '../../../data/epic_CF/references_epic_CF.p'
#GT_TAG_FEATS_PATH = '../../../data/epic_CF/gt_tag_feats_epic_CF.p'
#NUM_TAGS = 1188

#### CURRENT BREAKFAST FEATURES ####
#CORPUS_P_PATH = '../../data/breakfast_current/corpus_breakfast_current.p'
#REFERENCES_PATH = '../../data/breakfast_current/references_breakfast_current.p'
#GT_TAG_FEATS_PATH = '../../data/breakfast_current/gt_tag_feats_breakfast_current.p'
#NUM_TAGS = 47

#### FUTURE BREAKFAST FEATURES ####
#CORPUS_P_PATH = '../../data/breakfast_future/corpus_breakfast_future.p'
#REFERENCES_PATH = '../../data/breakfast_future/references_breakfast_future.p'
#GT_TAG_FEATS_PATH = '../../data/breakfast_future/gt_tag_feats_breakfast_future.p'
#NUM_TAGS = 47

#### CF BREAKFAST FEATURES ####
#CORPUS_P_PATH = '../../data/breakfast_CF/corpus_breakfast_CF.p'
#REFERENCES_PATH = '../../data/breakfast_CF/references_breakfast_CF.p'
#GT_TAG_FEATS_PATH = '../../data/breakfast_CF/gt_tag_feats_breakfast_CF.p'
#NUM_TAGS = 47

#### FUTURE RC BREAKFAST FEATURES ####
#CORPUS_P_PATH = '../../data/breakfast_futureRC/corpus_breakfast_futureRC.p'
#REFERENCES_PATH = '../../data/breakfast_futureRC/references_breakfast_futureRC.p'
#GT_TAG_FEATS_PATH = '../../data/breakfast_futureRC/gt_tag_feats_breakfast_futureRC.p'
#NUM_TAGS = 47

#### CF youcook2 FEATURES ####
#CORPUS_P_PATH = '../../../data/youcook2_CF/corpus_youcook2_CF.p'
#REFERENCES_PATH = '../../../data/youcook2_CF/references_youcook2_CF.p'
#GT_TAG_FEATS_PATH = '../../../data/youcook2_CF/gt_tag_feats_youcook2_CF.p'
#NUM_TAGS = 300

#### future youcook2 FEATURES ####
#CORPUS_P_PATH = '../../data/youcook2_future/corpus_youcook2_future.p'
#REFERENCES_PATH = '../../data/youcook2_future/references_youcook2_future.p'
#GT_TAG_FEATS_PATH = '../../data/youcook2_future/gt_tag_feats_youcook2_future.p'
#NUM_TAGS = 1839

#if __name__ == "__main__":
def step3_obtain_tags_1(config_obj, NUM_TAGS):
    CORPUS_P_PATH = config_obj.get('paths', 'corpus_p_path')
    REFERENCES_PATH = config_obj.get('paths', 'references_path')
    GT_TAG_FEATS_PATH = config_obj.get('paths', 'gt_tag_feats_path')

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

    print "3..Tags Obtained"
