import os
import csv
import json
import random
random.seed(42)

def load_csv(data_path):
    all_datas = []
    with open(data_path, 'r', encoding = 'utf-8') as read_f:
        reader = csv.DictReader(read_f)
        for row in reader:
            all_datas.append(row)
    return all_datas

def save_csv(data_path, datas, mode='w'):
    field_names = datas[0].keys()
    if not os.path.exists(data_path) or mode == 'w':
        with open(data_path, 'w', encoding='utf-8', newline='') as write_f:
            writer = csv.DictWriter(write_f, fieldnames=field_names)
            writer.writeheader()

    with open(data_path, 'a', encoding='utf-8', newline='') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        for data in datas:
            writer.writerow(data)

def load_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as read_f:
        datas = json.load(read_f)
    return datas

def process_sem16_dataset():
    SRC_DATASET_DIR = r'raw_datasets/Semeval16'
    DST_DATASET_DIR = r'dataset/Semeval16'
    train_valid_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'train.csv'))
    test_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'test.csv'))

    in_target_target_names = ['Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion', 'Atheism', 'Climate Change is a Real Concern']
    zero_shot_target_names = ['Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion', 'Atheism', 'Climate Change is a Real Concern', 'Donald Trump']

    # in-target datas
    for target_name in in_target_target_names:
        write_train_valid_datas = []
        write_test_datas = []
        for data in train_valid_datas:
            if data['Target'] == target_name:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': data['Target'],
                    'Stance': data['Stance']
                }
                write_train_valid_datas.append(write_data)
        for data in test_datas:
            if data['Target'] == target_name:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': data['Target'],
                    'Stance': data['Stance']
                }
                write_test_datas.append(write_data)
        
        random.shuffle(write_train_valid_datas)
        random.shuffle(test_datas)
        write_train_datas = write_train_valid_datas[:int(len(write_train_valid_datas)/0.8*0.7)]
        write_valid_datas = write_train_valid_datas[int(len(write_train_valid_datas)/0.8*0.7):]
        if not os.path.exists(f'{DST_DATASET_DIR}/in-target/{target_name}'):
            os.makedirs(f'{DST_DATASET_DIR}/in-target/{target_name}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_name}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_name}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_name}', f'test.csv'), write_test_datas)

    # zero-shot datas
    for target_name in zero_shot_target_names:
        write_all_datas = train_valid_datas + test_datas
        write_train_valid_datas = []
        write_test_datas = []
        for data in write_all_datas:
            write_data = {
                'Tweet': data['Tweet'],
                'Target': data['Target'],
                'Stance': data['Stance']
            }
            if data['Target'] != target_name:
                write_train_valid_datas.append(write_data)
            else:
                write_test_datas.append(write_data)
        
        random.shuffle(write_train_valid_datas)
        random.shuffle(write_test_datas)
        write_train_datas = write_train_valid_datas[:int(len(write_train_valid_datas)/0.8*0.7)]
        write_valid_datas = write_train_valid_datas[int(len(write_train_valid_datas)/0.8*0.7):]
        if not os.path.exists(f'{DST_DATASET_DIR}/zero-shot/{target_name}'):
            os.makedirs(f'{DST_DATASET_DIR}/zero-shot/{target_name}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_name}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_name}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_name}', f'test.csv'), write_test_datas)


def process_p_stance_dataset():
    SRC_DATASET_DIR = r'raw_datasets/P-Stance'
    DST_DATASET_DIR = r'datasets/P-Stance'

    in_target_target_names = ['Bernie Sanders', 'Joe Biden', 'Donald Trump']
    zero_shot_target_names = ['Bernie Sanders', 'Joe Biden', 'Donald Trump']

    target_name2data_path = {
        'Bernie Sanders': 'raw_%s_bernie.csv',
        'Joe Biden': 'raw_%s_biden.csv',
        'Donald Trump': 'raw_%s_trump.csv'
    }

    # in-target datas
    for target_name in in_target_target_names:
        train_datas = load_csv(os.path.join(SRC_DATASET_DIR, target_name2data_path[target_name] % 'train'))
        valid_datas = load_csv(os.path.join(SRC_DATASET_DIR, target_name2data_path[target_name] % 'val'))
        test_datas = load_csv(os.path.join(SRC_DATASET_DIR, target_name2data_path[target_name] % 'test'))

        write_train_datas = []
        for data in train_datas:
            if data['Stance'] == 'NONE':
                continue
            if data['Target'] == target_name:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': data['Target'],
                    'Stance': data['Stance']
                }
                write_train_datas.append(write_data)
        write_valid_datas = []
        for data in valid_datas:
            if data['Stance'] == 'NONE':
                continue
            if data['Target'] == target_name:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': data['Target'],
                    'Stance': data['Stance']
                }
                write_valid_datas.append(write_data)
        write_test_datas = []
        for data in test_datas:
            if data['Stance'] == 'NONE':
                continue
            if data['Target'] == target_name:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': data['Target'],
                    'Stance': data['Stance']
                }
                write_test_datas.append(write_data)

        if not os.path.exists(f'{DST_DATASET_DIR}/in-target/{target_name}'):
            os.makedirs(f'{DST_DATASET_DIR}/in-target/{target_name}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_name}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_name}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_name}', f'test.csv'), write_test_datas)

    # zero-shot datas
    all_datas = []
    for target_name in in_target_target_names:
        all_datas += load_csv(os.path.join(SRC_DATASET_DIR, target_name2data_path[target_name] % 'train'))
        all_datas += load_csv(os.path.join(SRC_DATASET_DIR, target_name2data_path[target_name] % 'val'))
        all_datas += load_csv(os.path.join(SRC_DATASET_DIR, target_name2data_path[target_name] % 'test'))
    
    for target_name in zero_shot_target_names:
        write_train_valid_datas = []
        write_test_datas = []
        for data in all_datas:
            if data['Stance'] == 'NONE':
                continue
            write_data = {
                'Tweet': data['Tweet'],
                'Target': data['Target'],
                'Stance': data['Stance']
            }
            if data['Target'] == target_name:
                write_test_datas.append(write_data)
            else:
                write_train_valid_datas.append(write_data)

        random.shuffle(write_train_valid_datas)
        random.shuffle(write_test_datas)
        write_train_datas = write_train_valid_datas[:int(len(write_train_valid_datas)/0.8*0.7)]
        write_valid_datas = write_train_valid_datas[int(len(write_train_valid_datas)/0.8*0.7):]
        if not os.path.exists(f'{DST_DATASET_DIR}/zero-shot/{target_name}'):
            os.makedirs(f'{DST_DATASET_DIR}/zero-shot/{target_name}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_name}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_name}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_name}', f'test.csv'), write_test_datas)

def process_covid_19_dataset():
    SRC_DATASET_DIR = r'raw_datasets/COVID-19'
    DST_DATASET_DIR = r'datasets/COVID-19'

    in_target_target_names_old = ['stay at home orders', 'school closures', 'fauci', 'face masks']
    in_target_target_names_old2new = {
        'stay at home orders': 'Stay at Home Orders',
        'school closures': 'Keeping Schools Closed',
        'fauci': 'Anthony S. Fauci, M.D.',
        'face masks': 'Wearing a Face Mask'
    }
    zero_shot_target_names_old = ['stay at home orders', 'school closures', 'fauci', 'face masks']
    zero_shot_target_names_old2new = {
        'stay at home orders': 'Stay at Home Orders',
        'school closures': 'Keeping Schools Closed',
        'fauci': 'Anthony S. Fauci, M.D.',
        'face masks': 'Wearing a Face Mask'
    }

    train_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'raw_train_all_onecol.csv'))
    valid_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'raw_val_all_onecol.csv'))
    test_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'raw_test_all_onecol.csv'))

    # in-target datas
    for target_name_old in in_target_target_names_old:
        write_train_datas = []
        for data in train_datas:
            if data['Target 1'] == target_name_old:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': in_target_target_names_old2new[data['Target 1']],
                    'Stance': data['Stance 1']
                }
                write_train_datas.append(write_data)
        write_valid_datas = []
        for data in valid_datas:
            if data['Target 1'] == target_name_old:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': in_target_target_names_old2new[data['Target 1']],
                    'Stance': data['Stance 1']
                }
                write_valid_datas.append(write_data)
        write_test_datas = []
        for data in test_datas:
            if data['Target 1'] == target_name_old:
                write_data = {
                    'Tweet': data['Tweet'],
                    'Target': in_target_target_names_old2new[data['Target 1']],
                    'Stance': data['Stance 1']
                }
                write_test_datas.append(write_data)

        if not os.path.exists(f'{DST_DATASET_DIR}/in-target/{in_target_target_names_old2new[target_name_old]}'):
            os.makedirs(f'{DST_DATASET_DIR}/in-target/{in_target_target_names_old2new[target_name_old]}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{in_target_target_names_old2new[target_name_old]}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{in_target_target_names_old2new[target_name_old]}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{in_target_target_names_old2new[target_name_old]}', f'test.csv'), write_test_datas)

    # zero-shot datas
    all_datas = train_datas + valid_datas + test_datas
    for target_name_old in zero_shot_target_names_old:
        write_train_valid_datas = []
        write_test_datas = []
        for data in all_datas:
            write_data = {
                'Tweet': data['Tweet'],
                'Target': zero_shot_target_names_old2new[data['Target 1']],
                'Stance': data['Stance 1']
            }
            if data['Target 1'] == target_name_old:
                write_test_datas.append(write_data)
            else:
                write_train_valid_datas.append(write_data)

        random.shuffle(write_train_valid_datas)
        random.shuffle(write_test_datas)
        write_train_datas = write_train_valid_datas[:int(len(write_train_valid_datas)/0.8*0.7)]
        write_valid_datas = write_train_valid_datas[int(len(write_train_valid_datas)/0.8*0.7):]
        if not os.path.exists(f'{DST_DATASET_DIR}/zero-shot/{zero_shot_target_names_old2new[target_name_old]}'):
            os.makedirs(f'{DST_DATASET_DIR}/zero-shot/{zero_shot_target_names_old2new[target_name_old]}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{zero_shot_target_names_old2new[target_name_old]}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{zero_shot_target_names_old2new[target_name_old]}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{zero_shot_target_names_old2new[target_name_old]}', f'test.csv'), write_test_datas)

def process_wtwt_dataset():
    SRC_DATASET_PATH = r'raw_datasets/wtwt/wtwt_process.json'
    DST_DATASET_DIR = r'datasets/WTWT'
    
    in_target_target_names_old = ['CVS_AET', 'CI_ESRX', 'ANTM_CI', 'AET_HUM', 'FOXA_DIS']
    zero_shot_target_names_old = ['CVS_AET', 'CI_ESRX', 'ANTM_CI', 'AET_HUM']
    target_map = {
        'CVS_AET': 'Merger and acquisition between CVS Health and Aetna',
        'CI_ESRX': 'Merger and acquisition between Cigna and Express Scripts',
        'ANTM_CI': 'Merger and acquisition between Anthem and Cigna',
        'AET_HUM': 'Merger and acquisition between Aetna and Humana',
        'FOXA_DIS': 'Merger and acquisition between Disney and 21st Century Fox'
    }

    all_datas = load_json(SRC_DATASET_PATH)
    # in-target datas
    for target_name_old in in_target_target_names_old:
        write_all_datas = []
        for data in all_datas:
            if data['merger'] == target_name_old:
                write_data = {
                    'Tweet': data['text'],
                    'Target': target_map[data['merger']],
                    'Stance': data['stance']
                }
                write_all_datas.append(write_data)

        random.shuffle(write_all_datas)
        write_train_datas = write_all_datas[:int(len(write_all_datas)*0.7)]
        write_valid_datas = write_all_datas[int(len(write_all_datas)*0.7):int(len(write_all_datas)*0.8)]
        write_test_datas = write_all_datas[int(len(write_all_datas)*0.8):]
        if not os.path.exists(f'{DST_DATASET_DIR}/in-target/{target_map[target_name_old]}'):
            os.makedirs(f'{DST_DATASET_DIR}/in-target/{target_map[target_name_old]}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_map[target_name_old]}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_map[target_name_old]}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/in-target/{target_map[target_name_old]}', f'test.csv'), write_test_datas)

    # zero-shot datas
    for target_name_old in zero_shot_target_names_old:
        write_train_valid_datas = []
        write_test_datas = []
        for data in all_datas:
            if data['merger'] not in zero_shot_target_names_old:
                continue
            write_data = {
                'Tweet': data['text'],
                'Target': target_map[data['merger']],
                'Stance': data['stance']
            }
            if data['merger'] == target_name_old:
                write_test_datas.append(write_data)
            else:
                write_train_valid_datas.append(write_data)

        random.shuffle(write_train_valid_datas)
        random.shuffle(write_test_datas)
        write_train_datas = write_train_valid_datas[:int(len(write_train_valid_datas)/0.8*0.7)]
        write_valid_datas = write_train_valid_datas[int(len(write_train_valid_datas)/0.8*0.7):]
        if not os.path.exists(f'{DST_DATASET_DIR}/zero-shot/{target_map[target_name_old]}'):
            os.makedirs(f'{DST_DATASET_DIR}/zero-shot/{target_map[target_name_old]}')
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_map[target_name_old]}', f'train.csv'), write_train_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_map[target_name_old]}', f'valid.csv'), write_valid_datas)
        save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot/{target_map[target_name_old]}', f'test.csv'), write_test_datas)

def process_vast_dataset():
    SRC_DATASET_DIR = r'raw_datasets/VAST'
    DST_DATASET_DIR = r'datasets/VAST'
    train_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'vast_train.csv'))
    valid_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'vast_dev.csv'))
    test_datas = load_csv(os.path.join(SRC_DATASET_DIR, 'vast_test.csv'))

    labelmap = {
        '0': 'con',
        '1': 'pro',
        '2': 'neutral'
    }

    # all 
    write_train_datas = []
    for data in train_datas:
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_train_datas.append(write_data)
    write_valid_datas = []
    for data in valid_datas:
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_valid_datas.append(write_data)
    write_test_datas = []
    for data in test_datas:
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_test_datas.append(write_data)
    if not os.path.exists(f'{DST_DATASET_DIR}/all'):
        os.makedirs(f'{DST_DATASET_DIR}/all')
    save_csv(os.path.join(f'{DST_DATASET_DIR}/all', f'train.csv'), write_train_datas)
    save_csv(os.path.join(f'{DST_DATASET_DIR}/all', f'valid.csv'), write_valid_datas)
    save_csv(os.path.join(f'{DST_DATASET_DIR}/all', f'test.csv'), write_test_datas)
    
    # zero-shot
    write_train_datas = []
    for data in train_datas:
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_train_datas.append(write_data)
    write_valid_datas = []
    for data in valid_datas:
        if data['seen?'] != '0':
            continue
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_valid_datas.append(write_data)
    write_test_datas = []
    for data in test_datas:
        if data['seen?'] != '0':
            continue
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_test_datas.append(write_data)
    if not os.path.exists(f'{DST_DATASET_DIR}/zero-shot'):
        os.makedirs(f'{DST_DATASET_DIR}/zero-shot')
    save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot', f'train.csv'), write_train_datas)
    save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot', f'valid.csv'), write_valid_datas)
    save_csv(os.path.join(f'{DST_DATASET_DIR}/zero-shot', f'test.csv'), write_test_datas)
    
    # few-shot
    write_train_datas = []
    for data in train_datas:
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_train_datas.append(write_data)
    write_valid_datas = []
    for data in valid_datas:
        if data['seen?'] != '1':
            continue
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_valid_datas.append(write_data)
    write_test_datas = []
    for data in test_datas:
        if data['seen?'] != '1':
            continue
        write_data = {
            'Tweet': data['post'],
            'Target': data['topic_str'],
            'Stance': labelmap[data['label']]
        }
        write_test_datas.append(write_data)
    if not os.path.exists(f'{DST_DATASET_DIR}/few-shot'):
        os.makedirs(f'{DST_DATASET_DIR}/few-shot')
    save_csv(os.path.join(f'{DST_DATASET_DIR}/few-shot', f'train.csv'), write_train_datas)
    save_csv(os.path.join(f'{DST_DATASET_DIR}/few-shot', f'valid.csv'), write_valid_datas)
    save_csv(os.path.join(f'{DST_DATASET_DIR}/few-shot', f'test.csv'), write_test_datas)


if __name__ == '__main__':
    # process_sem16_dataset()
    # process_p_stance_dataset()
    # process_covid_19_dataset()
    # process_wtwt_dataset()
    # process_vast_dataset()
    pass