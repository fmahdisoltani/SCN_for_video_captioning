"""Running script.
Usage:
  run.py <config_path>
  run.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

from docopt import docopt

#CONFIG_PATH = "./config_files/c1.yaml"

from yaml_config import YamlConfig
from preprocess.build_vocabulary_0 import step0_build_vocab
from preprocess.preprocess_1 import *
from preprocess.obtain_pretrained_word2vec_2 import *

from video_tagging.obtain_tags_1 import *
from video_tagging.training_video_tagging_model_2 import *

from model_scn_v2.SCN_training import *
from model_scn_v2.SCN_decode import *
from model_scn_v2.SCN_evaluation import *

if __name__ == "__main__":
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args["<config_path>"])

    # Run captioning model
    num_tags = step0_build_vocab(config_obj)
    print("num_tags is {}".format(num_tags))
#    step1_preprocess(config_obj)
#    step2_obtain_pretrained_word2vec(config_obj)
#    step3_obtain_tags_1(config_obj, num_tags)
#    step4_training_video_tagging_model(config_obj)
    step5_scn_train(config_obj)
    step6_scn_decode(config_obj)
    step7_scn_evaluate(config_obj)
