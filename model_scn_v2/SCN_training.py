'''
Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zg27@duke.edu, Sep., 2016
'''

import time
import logging
import cPickle
import h5py
import theano.tensor as tensor
import numpy as np
import scipy.io
import theano


from model_scn_v2.video_cap import init_params, init_tparams, build_model
from model_scn_v2.optimizers import Adam
from model_scn_v2.utils import get_minibatches_idx, zipp, unzip

IS_EPIC=False
#### CURRENT EPIC FEATURES ####
#HD5_NOUN_TRAIN = "/ais/fleet10/farzaneh/scn_captioning/data/epic_current/current_epic_noun_train.h5"
#HD5_NOUN_VALID = "/ais/fleet10/farzaneh/scn_captioning/data/epic_current/current_epic_noun_validation.h5"
#HD5_VERB_TRAIN = "/ais/fleet10/farzaneh/scn_captioning/data/epic_current/current_epic_verb_train.h5"
#HD5_VERB_VALID = "/ais/fleet10/farzaneh/scn_captioning/data/epic_current/current_epic_verb_validation.h5"
#TAG_FEATS_PRE_PATH = '../../data/epic_current/tag_feats_pred_epic_current.mat'
#CORPUS_P_PATH = "../../data/epic_current/corpus_epic_current.p"
#SAVE_TO_PATH = '../../data/epic_current/epic_current_result_scn_dropout.5_lr.0002_0.npz'
#N_WORDS = 1192
#DROP_OUT = 0.5
#LR = 0.0002
#MAX_EPOCH = 20 

#### FUTURE RC EPIC FEATURES ####
#HD5_NOUN_TRAIN = "/ais/fleet10/farzaneh/scn_captioning/data/epic_futureRC/future_epic_train_noun.h5"
#HD5_NOUN_VALID = "/ais/fleet10/farzaneh/scn_captioning/data/epic_futureRC/future_epic_val_noun.h5"
#HD5_VERB_TRAIN = "/ais/fleet10/farzaneh/scn_captioning/data/epic_futureRC/future_epic_train_verb.h5"
#HD5_VERB_VALID = "/ais/fleet10/farzaneh/scn_captioning/data/epic_futureRC/future_epic_val_verb.h5"
#TAG_FEATS_PRE_PATH = '../../data/epic_futureRC/tag_feats_pred_epic_future.mat'
#CORPUS_P_PATH = "../../data/epic_futureRC/corpus_epic_future.p"
#SAVE_TO_PATH = '../../data/epic_futureRC/epic_future_result_scn_dropout.5_lr.0002_0.npz'
#N_WORDS = 1188
#DROP_OUT = 0.5
#LR = 0.0002
#MAX_EPOCH = 20

#### CF EPIC FEATURES ####
#HD5_NOUN_TRAIN = '../../data/epic_CF/CF_epic_noun_train.h5'
#HD5_VERB_TRAIN = '../../data/epic_CF/CF_epic_verb_train.h5'
#HD5_NOUN_VALID = '../../data/epic_CF/CF_epic_noun_validation.h5'
#HD5_VERB_VALID = '../../data/epic_CF/CF_epic_verb_validation.h5'
#TAG_FEATS_PRE_PATH = '../../data/epic_CF/tag_feats_pred_epic_CF.mat'
#CORPUS_P_PATH = "../../data/epic_CF/corpus_epic_CF.p"
#SAVE_TO_PATH = '../../data/epic_CF/epic_CF_result_scn_dropout.8_lr.0002_0.npz'
#N_WORDS = 1188
#DROP_OUT = 0.8
#LR = 0.0002

#### CURRENT BREAKFAST FEATURES ####
#HD5_TRAIN = '../../data/breakfast_current/current_train_breakfast.h5'
#HD5_VALID = '../../data/breakfast_current/current_validation_breakfast.h5'
#TAG_FEATS_PRE_PATH = '../../data/breakfast_current/tag_feats_pred_breakfast_current.mat'
#CORPUS_P_PATH = "../../data/breakfast_current/corpus_breakfast_current.p"
#SAVE_TO_PATH = '../../data/breakfast_current/breakfast_current_result_scn_dropout.5_lr.0002_0.npz'
#N_WORDS = 47
#DROP_OUT = 0.5
#LR = 0.0002
#MAX_EPOCH = 20

#### FUTURE BREAKFAST FEATURES ####
#HD5_TRAIN = '../../data/breakfast_future/future_breakfast_train_all.h5'
#HD5_VALID = '../../data/breakfast_future/future_breakfast_val_all.h5'
#TAG_FEATS_PRE_PATH = '../../data/breakfast_future/tag_feats_pred_breakfast_future.mat'
#CORPUS_P_PATH = "../../data/breakfast_future/corpus_breakfast_future.p"
#SAVE_TO_PATH = '../../data/breakfast_future/breakfast_future_result_scn_dropout.5_lr.0002_0.npz'
#N_WORDS = 47
#DROP_OUT = 0.5
#LR = 0.0002
#MAX_EPOCH = 20

#### CF BREAKFAST FEATURES ####
#HD5_TRAIN = '../../data/breakfast_CF/CF_breakfast_train.h5'
#HD5_VALID = '../../data/breakfast_CF/CF_breakfast_val.h5'
#TAG_FEATS_PRE_PATH = '../../data/breakfast_CF/tag_feats_pred_breakfast_CF.mat'
#CORPUS_P_PATH = "../../data/breakfast_CF/corpus_breakfast_CF.p"
#SAVE_TO_PATH = '../../data/breakfast_CF/breakfast_CF_result_scn_dropout.5_lr.0002_epoch20.npz'
#N_WORDS = 47
#DROP_OUT = 0.5
#LR = 0.0002
#MAX_EPOCH = 20

#### CF EPIC FEATURES ####
#HD5_TRAIN_NOUN = '../../data/epic_CF/CF_epic_noun_train.h5'
#HD5_TRAIN_VERB = '../../data/epic_CF/CF_epic_verb_train.h5'
#HD5_VALID_NOUN = '../../data/epic_CF/CF_epic_noun_validation.h5'
#HD5_VALID_VERB = '../../data/epic_CF/CF_epic_verb_validation.h5'
#TAG_FEATS_PRE_PATH = '../../data/epic_CF/tag_feats_pred_epic_CF.mat'
#CORPUS_P_PATH = "../../data/epic_CF/corpus_epic_CF.p"
#SAVE_TO_PATH = 'epic_CF_result_scn.npz'
#N_WORDS = 1188

#### CF youcook2 FEATURES ####
#HD5_TRAIN = '../../data/youcook2_CF/CF_youcook2_currentSeg_train_all.h5'
#HD5_VALID = '../../data/youcook2_CF/CF_youcook2_currentSeg_val_all.h5'
#TAG_FEATS_PRE_PATH = '../../data/youcook2_CF/tag_feats_pred_youcook2_CF.mat'
#CORPUS_P_PATH = "../../data/youcook2_CF/corpus_youcook2_CF.p"
#SAVE_TO_PATH = '../../data/youcook2_CF/youcook2_CF_result_scn_dropout.8_lr.002_epoch30_300tags.npz'
#N_WORDS = 1839
#DROP_OUT = 0.8
#LR = 0.002
#MAX_EPOCH = 30

#### future youcook2 FEATURES ####
#HD5_TRAIN = '../../data/youcook2_future/future_youcook2_futureSeg_train_verb.h5'
#HD5_VALID = '../../data/youcook2_future/future_youcook2_futureSeg_val_verb.h5'
#TAG_FEATS_PRE_PATH = '../../data/youcook2_future/tag_feats_pred_youcook2_future.mat'
#CORPUS_P_PATH = "../../data/youcook2_future/corpus_youcook2_future.p"
#SAVE_TO_PATH = '../../data/youcook2_future/youcook2_future_result_scn_dropout.5_lr.0002_epoch20_0.npz'
#N_WORDS = 1839
#DROP_OUT = 0.5
#LR = 0.0002
#MAX_EPOCH = 20

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

def load_hd5(hd5_name, keyword):
    #for epic datasets, keyword is either noun or verb
    with h5py.File(hd5_name,"r") as f:
        for key in f.keys():
           print(key)
        grid = f[keyword][()] #Convert to numpy
    #feats= np.squeeze(grid, axis=1)
    feats = grid
    return feats

def prepare_data(seqs):
    
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.
    return x, x_mask

def calu_negll(f_cost, prepare_data, data, img_feats, tag_feats, iterator):

    totalcost = 0.
    totallen = 0.
    for _, valid_index in iterator:
        x = [data[0][t]for t in valid_index]
        x, mask = prepare_data(x)
        y = np.array([tag_feats[data[1][t]]for t in valid_index])
        z = np.array([img_feats[data[1][t]]for t in valid_index])
                
        length = np.sum(mask)
        cost = f_cost(x, mask,y,z) * x.shape[1]
        totalcost += cost
        totallen += length
    return totalcost/totallen


""" Training the model. """

def train_model(logger, train, valid, test, img_feats, tag_feats, W, n_words=-1, n_x=300, n_h=512,
    n_f=512, max_epochs=-1, lrate=-1, batch_size=640, valid_batch_size=64, 
    dropout_val=-1, dispFreq=10, validFreq=200, saveFreq=1000, 
    saveto = -1):
        
    """ n_words : vocabulary size
        n_x : word embedding dimension
        n_h : LSTM/GRU number of hidden units 
        n_f : number of factors used in SCN
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dropout_val : the probability of dropout
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        saveFreq : save results after this number of update.
        saveto : where to save.
    """

    options = {}
    options['n_words'] = n_words
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['n_f'] = n_f
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
    
    options['n_z'] = img_feats.shape[1]
    options['n_y'] = tag_feats.shape[1]
    options['SEED'] = SEED
    logger.info('Model options {}'.format(options))
    logger.info('{} train examples'.format(len(train[0])))
    logger.info('{} valid examples'.format(len(valid[0])))
    logger.info('{} test examples'.format(len(test[0])))

    logger.info('Building model...')
    
    params = init_params(options,W)
    tparams = init_tparams(params)

    (use_noise, x, mask, y, z, cost) = build_model(tparams,options)
    
    f_cost = theano.function([x, mask, y, z], cost, name='f_cost')
    
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, [x, mask, y, z], lr)

    logger.info('Training model...')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    
    estop = False  # early stop
    history_negll = []
    best_p = None
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    try:
        for eidx in xrange(max_epochs):
            print "Epoch number: {}".format(eidx)
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(dropout_val)

                x = [train[0][t]for t in train_index]
                y = np.array([tag_feats[train[1][t]]for t in train_index])
                z = np.array([img_feats[train[1][t]]for t in train_index])
                x, mask = prepare_data(x)
                cost = f_grad_shared(x, mask,y,z)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    logger.info('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, cost))
                    
                if np.mod(uidx, saveFreq) == 0:
                    logger.info('Saving ...')
                
                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_negll=history_negll, **params)
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    
                    #train_negll = calu_negll(f_cost, prepare_data, train, img_feats, kf)
                    valid_negll = calu_negll(f_cost, prepare_data, valid, img_feats, tag_feats, kf_valid)
                    test_negll = calu_negll(f_cost, prepare_data, test, img_feats, tag_feats, kf_test)
                    history_negll.append([valid_negll, test_negll])
                    
                    if (uidx == 0 or
                        valid_negll <= np.array(history_negll)[:,0].min()):
                             
                        best_p = unzip(tparams)
                        bad_counter = 0
                        
                    logger.info('Perp: Valid {} Test {}'.format(np.exp(valid_negll), np.exp(test_negll)))

                    if (len(history_negll) > 10 and
                        valid_negll >= np.array(history_negll)[:-10,0].min()):
                            bad_counter += 1
                            if bad_counter > 10:
                                logger.info('Early Stop!')
                                estop = True
                                break

            if estop:
                print ("P"*100)
                break

    except KeyboardInterrupt:
        logger.info('Training interupted')

    end_time = time.time()
    
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
        
    use_noise.set_value(0.)
    #kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    #train_negll = calu_negll(f_cost, prepare_data, train, img_feats, kf_train_sorted)
    valid_negll = calu_negll(f_cost, prepare_data, valid, img_feats, tag_feats, kf_valid)
    test_negll = calu_negll(f_cost, prepare_data, test, img_feats, tag_feats, kf_test)
    
    logger.info('Final Results...')
    logger.info('Perp: Valid {} Test {}'.format(np.exp(valid_negll), np.exp(test_negll)))
    np.savez(saveto, history_negll=history_negll, **best_p)

    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    return valid_negll, test_negll

#if __name__ == '__main__':
def step5_scn_train(config_obj):
    # https://docs.python.org/2/howto/logging-cookbook.html
    dropout = config_obj.get("params","dropout")
    lr = config_obj.get("params","lr")
    max_epochs = config_obj.get("params","max_epochs")
    CORPUS_P_PATH =config_obj.get("paths", "corpus_p_path")
    HD5_TRAIN = config_obj.get("paths", "hd5_train")
    HD5_VALID = config_obj.get("paths", "hd5_valid")
    TAG_FEATS_PRED = config_obj.get("paths", "tag_feats_pred_path")
    SAVE_TO_PATH = config_obj.get("paths", "save_to_path")

    logger = logging.getLogger('eval_youtube_scn')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_youtube_scn.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
        
    x = cPickle.load(open(CORPUS_P_PATH,"rb"))
    train, val, test = x[0], x[1], x[2]
    wordtoix, ixtoword = x[3], x[4]
    W = x[5]
    del x
    n_words = len(ixtoword)
    
    #data = scipy.io.loadmat('./data/youtube2text/c3d_feats.mat')
    #c3d_img_feats = data['feats'].astype(theano.config.floatX)

    #data = scipy.io.loadmat('./data/youtube2text/resnet_feats.mat')
    #resnet_img_feats = data['feature'].T.astype(theano.config.floatX)
    
#    hdf5_name = "/ais/fleet10/farzaneh/scn_captioning/SCN_for_video_captioning/data/sample/features_epic/epic_features_epic_train.h5"
#    with h5py.File(hdf5_name,"r") as f:
#        for key in f.keys():
#           print(key)
#
#        grid = f["features"][()] #Convert to numpy
#    #print(grid[0,:])
#    grid2= np.squeeze(grid, axis=1) 
    #img_feats = np.concatenate((c3d_img_feats,resnet_img_feats),axis=1)
#    img_feats = grid2
   
    if IS_EPIC: 
        train_noun_feats = load_hd5(HD5_NOUN_TRAIN, "noun")
        train_verb_feats = load_hd5(HD5_VERB_TRAIN, "verb")
        img_feats = np.concatenate([train_noun_feats,train_verb_feats], axis=1)   #[train_noun_feats, train_verb_feats]    
        ##img_feats = train_verb_feats

        valid_noun_feats= load_hd5(HD5_NOUN_VALID, "noun")
        valid_verb_feats = load_hd5(HD5_VERB_VALID, "verb")
        img_feats_valid = np.concatenate([valid_noun_feats,valid_verb_feats], axis=1)   #[train_noun_feats, train_verb_feats]    
        ##img_feats_valid = valid_noun_feats
    else:
        img_feats = load_hd5(HD5_TRAIN, "all")
        img_feats_valid = load_hd5(HD5_VALID, "all")

    data = scipy.io.loadmat(TAG_FEATS_PRED)
    tag_feats = data['feats'].astype(theano.config.floatX)
    [val_negll, te_negll] = train_model(logger, train, val, test, img_feats, tag_feats, W,
                             dropout_val=dropout, max_epochs=max_epochs, lrate=lr, n_words = n_words, 
                            saveto= SAVE_TO_PATH)
        
