import os
import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        # 각 key에 대해 재귀적으로 변환합니다.
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

current_dir = os.path.dirname(__file__)
config_file = os.path.join(current_dir, 'config.yaml')
with open(config_file, 'r') as file:
    config_dict = yaml.safe_load(file)

config = dict_to_namespace(config_dict)