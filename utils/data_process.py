import csv
import json
import os
import random

def set_random_seed():
    random.seed(667)

field_names = ['tweet_id', 'tweet_text', 'target', 'label']
train_proportion = [0, 0.7]
valid_proportion = [0.7, 0.8]
test_proportion = [0.8, 1]

def read_csv(path):
    all_datas = []
    with open(path, 'r', encoding='utf-8') as read_f:
        reader = csv.DictReader(read_f)
        for row in reader:
            all_datas.append(row)
    return all_datas

def write_csv(path, datas, field_names):
    with open(path, 'w', newline='', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        writer.writeheader()
        for data in datas:
            writer.writerow(data)

semeval16_data_path = 'dataset/processed_dataset/Semeval16/semeval16.csv'
semeval16_data_dir = 'dataset/processed_dataset/Semeval16'
def semeval16_data_process():
    data_dir = 'dataset/processed_dataset/Semeval16'
    data_dirs = []
    for file_name in os.listdir(data_dir):
        if os.path.isdir(data_dir + '/' + file_name):
            data_dirs.append(data_dir + '/' + file_name)
    data_paths = [file_name + '/test.csv' for file_name in data_dirs]

    all_datas = []
    for data_path in data_paths:
        with open(data_path, 'r', encoding='utf-8') as read_f:
            reader = csv.DictReader(read_f)
            for row in reader:
                all_datas.append(row)
    set_random_seed()
    random.shuffle(all_datas)

    with open(semeval16_data_path, 'w', newline='', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        writer.writeheader()
        for data in all_datas:
            writer.writerow(data)

    train_datas = all_datas[int(len(all_datas) * train_proportion[0]) : int(len(all_datas) * train_proportion[1])]
    valid_datas = all_datas[int(len(all_datas) * valid_proportion[0]) : int(len(all_datas) * valid_proportion[1])]
    test_datas = all_datas[int(len(all_datas) * test_proportion[0]) : int(len(all_datas) * test_proportion[1])]

    with open(semeval16_data_dir + '/semeval16_train.csv', 'w', newline='', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        writer.writeheader()
        for data in train_datas:
            writer.writerow(data)

    with open(semeval16_data_dir + '/semeval16_valid.csv', 'w', newline='', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        writer.writeheader()
        for data in valid_datas:
            writer.writerow(data)

    with open(semeval16_data_dir + '/semeval16_test.csv', 'w', newline='', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        writer.writeheader()
        for data in test_datas:
            writer.writerow(data)

p_stance_dir = 'dataset/processed_dataset/P-Stance'
def p_stance_process():
    train_datas = []
    valid_datas = []
    test_datas = []
    for file_name in os.listdir(p_stance_dir):
        if 'raw_train' in file_name:
            for data in read_csv(p_stance_dir + '/' + file_name):
                if data['label'] == 'NONE': # 原论文中仅二分类
                    continue
                train_datas.append(data)
        elif 'raw_val' in file_name:
            for data in read_csv(p_stance_dir + '/' + file_name):
                if data['label'] == 'NONE': # 原论文中仅二分类
                    continue
                valid_datas.append(data)
        elif 'raw_test' in file_name:
            for data in read_csv(p_stance_dir + '/' + file_name):
                if data['label'] == 'NONE': # 原论文中仅二分类
                    continue
                test_datas.append(data)

    set_random_seed()
    random.shuffle(train_datas)
    set_random_seed()
    random.shuffle(valid_datas)
    set_random_seed()
    random.shuffle(test_datas)
    all_datas = train_datas + valid_datas + test_datas

    # write_csv(p_stance_dir + '/p-stance.csv', all_datas)
    write_csv(p_stance_dir + '/p-stance_train.csv', train_datas)
    write_csv(p_stance_dir + '/p-stance_valid.csv', valid_datas)
    write_csv(p_stance_dir + '/p-stance_test.csv', test_datas)


covid19_train_data_path = 'dataset/processed_dataset/COVID-19/raw_train_all_onecol.csv'
covid19_valid_data_path = 'dataset/processed_dataset/COVID-19/raw_val_all_onecol.csv'
covid19_test_data_path = 'dataset/processed_dataset/COVID-19/raw_test_all_onecol.csv'
covid19_write_data_dir = 'dataset/processed_dataset/COVID-19'

def covid19_data_process():
    train_datas = read_csv(covid19_train_data_path)
    valid_datas = read_csv(covid19_valid_data_path)
    test_datas = read_csv(covid19_test_data_path)
    set_random_seed()
    random.shuffle(train_datas)
    set_random_seed()
    random.shuffle(valid_datas)
    set_random_seed()
    random.shuffle(test_datas)
    all_datas = train_datas + valid_datas + test_datas

    write_csv(covid19_write_data_dir + '/covid19.csv', all_datas)
    write_csv(covid19_write_data_dir + '/covid19_train.csv', train_datas)
    write_csv(covid19_write_data_dir + '/covid19_valid.csv', valid_datas)
    write_csv(covid19_write_data_dir + '/covid19_test.csv', test_datas)