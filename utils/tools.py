import yaml
import random
import torch
import numpy as np
import csv
import os

def read_yaml(file_name):
    with open(file_name, 'r', encoding='utf8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def set_seed(seed_num=42):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def load_csv(data_path):
    all_datas = []
    with open(data_path, 'r', encoding='utf-8') as read_f:
        reader = csv.DictReader(read_f)
        for row in reader:
            all_datas.append(row)
    return all_datas

def save_csv(data_path, datas, mode='w'):
    field_names = datas[0].keys()
    if not os.path.exists(data_path) or mode == 'w':
        with open(data_path, 'w', encoding='utf-8') as write_f:
            writer = csv.DictWriter(write_f, fieldnames=field_names)
            writer.writeheader()

    with open(data_path, 'a', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        for data in datas:
            writer.writerow(data)