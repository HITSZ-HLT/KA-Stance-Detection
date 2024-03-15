import os
import sys
sys.path.append('./')
from data_config import data_configs
import csv
csv.field_size_limit(500 * 1024 * 1024)
from utils.tools import load_csv, save_csv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--process_dataset_for_knowledge', action='store_true')
parser.add_argument('--get_kasd_dataset', action='store_true')
GLOBAL_ARGS = parser.parse_args()

dataset_names = [
    'sem16',
    'p_stance',
    'covid_19',
    'vast',
]

def process_dataset_for_knowledge():
    write_dataset = 'datasets/processed_dataset_all'
    dataset_file_paths = {
        'sem16': 'Semeval16/all_data.csv',
        'p_stance': 'P-Stance/all_data.csv',
        'covid_19': 'COVID-19/all_data.csv',
        'vast': 'VAST/all_data.csv',
    }
    for dataset_name in dataset_names[:-1]:
        all_datas = []
        for target_name in set(data_configs[dataset_name].in_target_target_names + data_configs[dataset_name].zero_shot_target_names):
            all_datas += load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/test.csv")
        if not os.path.exists(os.path.dirname(f"{write_dataset}/{dataset_file_paths[dataset_name]}")):
            os.makedirs(os.path.dirname(f"{write_dataset}/{dataset_file_paths[dataset_name]}"))
        save_csv(f"{write_dataset}/{dataset_file_paths[dataset_name]}", all_datas)
    for dataset_name in dataset_names[-1:]:
        all_datas = []
        for target_name in data_configs[dataset_name].zero_shot_target_names:
            all_datas += load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/train.csv")
            all_datas += load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/valid.csv")
            all_datas += load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/test.csv")
        if not os.path.exists(os.path.dirname(f"{write_dataset}/{dataset_file_paths[dataset_name]}")):
            os.makedirs(os.path.dirname(f"{write_dataset}/{dataset_file_paths[dataset_name]}")) 
        save_csv(f"{write_dataset}/{dataset_file_paths[dataset_name]}", all_datas)

def get_kasd_dataset():
    read_episodic_knowledge_dir = 'datasets/episodic_knowledge'
    read_discourse_knowledge_dir = 'datasets/discourse_knowledge'
    dataset_file_paths = {
        'sem16': 'Semeval16/all_data.csv',
        'p_stance': 'P-Stance/all_data.csv',
        'covid_19': 'COVID-19/all_data.csv',
        'vast': 'VAST/all_data.csv',
    }
    for dataset_name in dataset_names[:-1]:
        all_episodic_knowledge_datas = load_csv(f"{read_episodic_knowledge_dir}/{dataset_file_paths[dataset_name]}")
        all_discourse_knowledge_datas = load_csv(f"{read_discourse_knowledge_dir}/{dataset_file_paths[dataset_name]}")
        tweet2episodic_knowledgeidx = {f"{data['Target']}:{data['Tweet']}": idx for idx, data in enumerate(all_episodic_knowledge_datas)}
        tweet2discourse_knowledgeidx = {f"{data['Target']}:{data['Tweet']}": idx for idx, data in enumerate(all_discourse_knowledge_datas)}
        # in-target
        for target_name in data_configs[dataset_name].in_target_target_names:
            if not os.path.exists(f"{data_configs[dataset_name].in_target_kasd_data_dir}/{target_name}"):
                os.makedirs(f"{data_configs[dataset_name].in_target_kasd_data_dir}/{target_name}")
            train_datas = load_csv(f"{data_configs[dataset_name].in_target_data_dir}/{target_name}/train.csv")
            for data in train_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].in_target_kasd_data_dir}/{target_name}/train.csv", train_datas)
            valid_datas = load_csv(f"{data_configs[dataset_name].in_target_data_dir}/{target_name}/valid.csv")
            for data in valid_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].in_target_kasd_data_dir}/{target_name}/valid.csv", valid_datas)
            test_datas = load_csv(f"{data_configs[dataset_name].in_target_data_dir}/{target_name}/test.csv")
            for data in test_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].in_target_kasd_data_dir}/{target_name}/test.csv", test_datas)
        
        # zero-shot
        for target_name in data_configs[dataset_name].zero_shot_target_names:
            if not os.path.exists(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}"):
                os.makedirs(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}")
            train_datas = load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/train.csv")
            for data in train_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}/train.csv", train_datas)
            valid_datas = load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/valid.csv")
            for data in valid_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}/valid.csv", valid_datas)
            test_datas = load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/test.csv")
            for data in test_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}/test.csv", test_datas)

    for dataset_name in dataset_names[-1:]:
        all_episodic_knowledge_datas = load_csv(f"{read_episodic_knowledge_dir}/{dataset_file_paths[dataset_name]}")
        all_discourse_knowledge_datas = load_csv(f"{read_discourse_knowledge_dir}/{dataset_file_paths[dataset_name]}")
        tweet2episodic_knowledgeidx = {f"{data['Target']}:{data['Tweet']}": idx for idx, data in enumerate(all_episodic_knowledge_datas)}
        tweet2discourse_knowledgeidx = {f"{data['Target']}:{data['Tweet']}": idx for idx, data in enumerate(all_discourse_knowledge_datas)}
        # zero-shot
        for target_name in data_configs[dataset_name].zero_shot_target_names:
            if not os.path.exists(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}"):
                os.makedirs(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}")
            train_datas = load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/train.csv")
            for data in train_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}/train.csv", train_datas)
            valid_datas = load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/valid.csv")
            for data in valid_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}/valid.csv", valid_datas)
            test_datas = load_csv(f"{data_configs[dataset_name].zero_shot_data_dir}/{target_name}/test.csv")
            for data in test_datas:
                data['episodic_knowledge'] = all_episodic_knowledge_datas[tweet2episodic_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['episodic_knowledge']
                data['discourse_knowledge'] = all_discourse_knowledge_datas[tweet2discourse_knowledgeidx[f"{data['Target']}:{data['Tweet']}"]]['discourse_knowledge']
            save_csv(f"{data_configs[dataset_name].zero_shot_kasd_data_dir}/{target_name}/test.csv", test_datas)

if __name__ == '__main__':
    if GLOBAL_ARGS.process_dataset_for_knowledge:
        process_dataset_for_knowledge()
    if GLOBAL_ARGS.get_kasd_dataset:
        get_kasd_dataset()