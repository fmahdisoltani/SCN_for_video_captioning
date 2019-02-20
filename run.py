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
import SCN_training, SCN_decode, SCN_evaluation
from preprocess.build_vocabulary_0 import step0_build_vocab

if __name__ == "__main__":
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args["<config_path>"])

    # Run captioning model
    step0_build_vocab(config_obj)

