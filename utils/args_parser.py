import yaml
import os

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return  path


def newest(path):
    if not os.path.exists(path):
        return None
    files = os.listdir(path)
    if len(files) == 0:
        return None
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)



def save_yaml(yaml_object, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(yaml_object, yaml_file, default_flow_style=False)

    print(f'Saving yaml: {file_path}')
    return

def parse_args(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return cfg



def flatten_cfg(cfg):
    cfg_flat = {}
    for key, value in cfg.items():
        if not isinstance(value, dict):
            cfg_flat[key] = value
        else:
            for key2, value2 in value.items():
                if not isinstance(value2, dict):
                    cfg_flat[f'{key}_{key2}'] = value2
                else:
                    for key3, value3 in value2.items():
                        cfg_flat[f'{key}_{key2}_{key3}'] = value3

    return cfg_flat

