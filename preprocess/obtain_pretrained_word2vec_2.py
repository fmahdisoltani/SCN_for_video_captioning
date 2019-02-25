import numpy as np
import cPickle

#### CURRENT EPIC FEATURES ####
#DATA_P_PATH = '../../../data/epic_current/data_epic_current.p'
#CORPUS_P_PATH = '../../../data/epic_current/corpus_epic_current.p'

#### FUTURE EPIC FEATURES ####
#DATA_P_PATH = '../../../data/epic_futureRC/data_epic_future.p'
#CORPUS_P_PATH = '../../../data/epic_futureRC/corpus_epic_future.p'

#### CF EPIC FEATURES ####
#DATA_P_PATH = '../../../data/epic_CF/data_epic_CF.p'
#CORPUS_P_PATH = '../../../data/epic_CF/corpus_epic_CF.p'

#### CURRENT BREAKFAST FEATURES ####
#DATA_P_PATH = '../../data/breakfast_current/data_breakfast_current.p'
#CORPUS_P_PATH = '../../data/breakfast_current/corpus_breakfast_current.p'

#### FUTURE BREAKFAST FEATURES ####
#DATA_P_PATH = '../../data/breakfast_future/data_breakfast_future.p'
#CORPUS_P_PATH = '../../data/breakfast_future/corpus_breakfast_future.p'

#### CF BREAKFAST FEATURES ####
#DATA_P_PATH = '../../data/breakfast_CF/data_breakfast_CF.p'
#CORPUS_P_PATH = '../../data/breakfast_CF/corpus_breakfast_CF.p'

#### FUTURE RC BREAKFAST FEATURES ####
#DATA_P_PATH = '../../data/breakfast_futureRC/data_breakfast_futureRC.p'
#CORPUS_P_PATH = '../../data/breakfast_futureRC/corpus_breakfast_futureRC.p'

#### CF youcook2 FEATURES ####
#DATA_P_PATH = '../../data/youcook2_CF/data_youcook2_CF.p'
#CORPUS_P_PATH = '../../data/youcook2_CF/corpus_youcook2_CF.p'

#### future youcook2 FEATURES ####
#DATA_P_PATH = '../../data/youcook2_future/data_youcook2_future.p'
#CORPUS_P_PATH = '../../data/youcook2_future/corpus_youcook2_future.p'

#GoogleNews_PATH = '../../data/GoogleNews-vectors-negative300.bin'

def get_W(w2v, word2idx, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(w2v)
    W = np.zeros(shape=(vocab_size, k))            
     
    for word in w2v:
        W[word2idx[word]] = w2v[word]
     
    return W
 
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs
 
def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  
 
def step2_obtain_pretrained_word2vec(config_obj):
#if __name__=="__main__":   i
    DATA_P_PATH = config_obj.get('paths', 'data_p_path')
    CORPUS_P_PATH = config_obj.get('paths', 'corpus_p_path')
    GoogleNews_PATH = config_obj.get('paths', 'google_news_path')
   
    w2v_file = GoogleNews_PATH
     
    x = cPickle.load(open(DATA_P_PATH,"rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    del x
    n_words = len(ixtoword)
       
    w2v = load_bin_vec(w2v_file, wordtoix)
    add_unknown_words(w2v, wordtoix)
    W = get_W(w2v,wordtoix)
     
    #rand_vecs = {}
    #add_unknown_words(rand_vecs, wordtoix)
    #W2 = get_W(rand_vecs,wordtoix)
     
    cPickle.dump([train, val, test, wordtoix, ixtoword, W], open(CORPUS_P_PATH, "wb"))
    print "3..Pretrained word vector created!"
