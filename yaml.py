import os
import yaml


class YamlConfig(object):

    def __init__(self, path=None, config_dict=None):
        self.config_dict = {} if config_dict is None else config_dict
        if path:
            self.parse(path)

    def parse(self, path):
        with open(path, "r") as f:
            self.config_dict.update(yaml.load(f.read()))

    def get(self, *keys):
        output = self.config_dict
        for key in keys:
            output = output[key]
        return output

    def save(self, folder, filename="config.yaml"):
        with open(os.path.join(folder, filename), "w") as f:

