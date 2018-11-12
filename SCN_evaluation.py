"""
Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zg27@duke.edu, Sep., 2016

Computes the BLEU, ROUGE, METEOR, and CIDER
using the COCO metrics scripts
"""

# this requires the coco-caption package, https://github.com/tylin/coco-caption
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import cPickle

#### CURRENT BREAKFAST FEATURES ####
#REFERENCES_PATH = "../../data/breakfast_current/references_breakfast_current.p"
#TEST_RESULTS = '../../data/breakfast_current/breakfast_current_scn_test_dropout.5_lr.0002.txt'

#### FUTURE BREAKFAST FEATURES ####
#REFERENCES_PATH = "../../data/breakfast_future/references_breakfast_future.p"
#TEST_RESULTS = '../../data/breakfast_future/breakfast_future_scn_test_dropout.5_lr.0002.txt'

#### CF BREAKFAST FEATURES ####
#REFERENCES_PATH = "../../data/breakfast_CF/references_breakfast_CF.p"
#TEST_RESULTS = '../../data/breakfast_CF/breakfast_CF_scn_test_dropout.5_lr.0002.txt'

#### CF youcook2 FEATURES ####
#REFERENCES_PATH = "../../data/youcook2_CF/references_youcook2_CF.p"
#TEST_RESULTS = '../../data/youcook2_CF/youcook2_CF_scn_test_dropout.8_lr.002.txt'

#### future youcook2 FEATURES ####
REFERENCES_PATH = "../../data/youcook2_future/references_youcook2_future.p"
TEST_RESULTS = '../../data/youcook2_future/youcook2_future_scn_test_dropout.5_lr.0002.txt' 

#### CF EPIC FEATURES ####
#REFERENCES_PATH = "../../data/epic_CF/references_epic_CF.p"
#TEST_RESULTS = '../../data/epic_CF/epic_CF_scn_test_dropout.8_lr.0002.txt'

#### FUTURE EPIC FEATURES ####
#REFERENCES_PATH = "../../data/epic_futureRC/references_epic_future.p"
#TEST_RESULTS = '../../data/epic_futureRC/epic_future_scn_test_dropout.5_lr.0002.txt'

#### CURRENT EPIC FEATURES ####
#REFERENCES_PATH = "../../data/epic_current/references_epic_current.p"
#TEST_RESULTS = '../../data/epic_current/epic_current_scn_test_dropout.5_lr.0002.txt'


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

if __name__ == '__main__':
    
    # this is the generated captions
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(open(TEST_RESULTS , 'rb') )}  
    
    # this is the ground truth captions
    x = cPickle.load(open(REFERENCES_PATH,"rb"))
    refs = x[2]
    del x
    
    refs = {idx: ref for (idx, ref) in enumerate(refs)}
    
    print score(refs, hypo)
    
    
    
    
        

