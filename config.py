import os
import pathlib
import json
import numpy as np

"""
Put this file inside the video data folder
"""

def read_config(filepath):
    try:
        with open(filepath, 'r') as f:
            cfg = json.load(f)
            return cfg
    except:
        return None


def get_config(read_cfg=True):
    dataset = pathlib.Path(__file__).parent.absolute()
    if read_cfg:
        cfg_path = os.path.join(dataset, 'config.json')
        cfg = read_config(cfg_path)
        if cfg is not None:
            print(f"Successfully load data config from {cfg_path}.")
            return cfg
        else:
            print(f"Fail to load config from {cfg_path}. Create a new one instead.")

    cfg = dict()
    cfg['dataset'] = dataset
    cfg['id'] = 'monday_roll15'
    cfg['hsv_ranges'] = {
        'red': [[160, 100, 50], [180, 255, 255]],
        'green': [[75, 80, 25], [100, 255, 255]],
        'blue': [[100, 50, 50], [130, 255, 255]],
    }
    cfg["cam_intr"] = np.array([
        [ 902.5143432617188, 0, 648.3026123046875 ],
        [ 0, 902.5889892578125, 372.9092712402344 ],
        [ 0, 0, 1 ]
    ])
    cfg['im_w'] = 1280
    cfg['im_h'] = 720
    cfg['depth_scale'] = 4000
    cfg['num_rods'] = 3
    cfg['rod_length'] = 0.36
    cfg['rod_diameter'] = 0.02
    cfg['rod_scale'] = 1
    cfg['depth_trunc'] = 3
    cfg['cable_length_scale'] = 1000
    cfg['end_cap_colors'] = ['red', 'green', 'blue']
    cfg['node_to_color'] = ['red', 'red', 'green', 'green', 'blue', 'blue']
    cfg['color_to_rod'] = {
        'red': [0, 1],
        'green': [2, 3],
        'blue': [4, 5]
    }
    cfg['sensor_to_tendon'] = {
        0: [3, 5],
        1: [1, 3],
        2: [1, 5],
        3: [0, 2],
        4: [0, 4],
        5: [2, 4],
        6: [2, 5],
        7: [0, 3],
        8: [1, 4]
    }
    cfg['sensor_status'] = {
        0: True,
        1: True,
        2: True,
        3: True,
        4: True,
        5: True,
        6: True,
        7: True,
        8: True
    }
    return cfg


def write_config(cfg):
    new_cfg = dict()
    for key, value in cfg.items():
        if isinstance(value, np.ndarray):
            new_cfg[key] = value.tolist()
        elif isinstance(value, pathlib.PosixPath):
            new_cfg[key] = str(value)
        else:
            new_cfg[key] = value

    filepath = os.path.join(pathlib.Path(__file__).parent.absolute(), 'config.json')
    with open(filepath, 'w') as f:
        json.dump(new_cfg, f, indent=4)
