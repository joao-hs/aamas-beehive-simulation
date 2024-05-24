import json
import os



def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def get_config():
    # read environment variable to get the path to the config file
    config_path = os.getenv('CONFIG_PATH')
    return read_config(config_path)
