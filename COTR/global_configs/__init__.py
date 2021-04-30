import os
import json

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(__location__, 'dataset_config.json'), 'r') as f:
    dataset_config = json.load(f)
with open(os.path.join(__location__, 'commons.json'), 'r') as f:
    general_config = json.load(f)
assert os.path.isdir(general_config['out']), f'Please create {general_config["out"]}'
assert os.path.isdir(general_config['tb_out']), f'Please create {general_config["tb_out"]}'
