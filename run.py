"""Running script.
Usage:
  run.py <config_path>
  run.py (-h | --help)

Options:
  <configpath>           Path to a config file.
  -h --help              Show this screen.
"""

from docopt import docopt

from SCN_for_video_captioning import SCN_training, SCN_decode, SCN_evaluation


if __name__ == "__main__":
    # Get argument
    args = docopt(__doc__)

    # Build a dictionary that contains fields of config file
    config_obj = YamlConfig(args["<config_path>"])

    # Run captioning model
    train_model(config_obj)

