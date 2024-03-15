import re
import os
import csv
import json
import time
import logging
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
import requests
import sys
sys.path.append('./')
from kasd_knowledge.prompts import discoures_knowledge_prompt

api_key = '***'
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

write_file_dir = 'datasets/discourse_knowledge'
if not os.path.exists(f'{write_file_dir}/logs'):
    os.makedirs(f'{write_file_dir}/logs')
fh = logging.FileHandler(f'{write_file_dir}/logs/' + 'openai_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +'.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

dataset_names = [
    'sem16',
    'p_stance',
    'covid_19',
    'vast',
]

read_data_dir = 'datasets/processed_dataset_all'
dataset_file_paths = {
    'sem16': 'Semeval16/all_data.csv',
    'p_stance': 'P-Stance/all_data.csv',
    'covid_19': 'COVID-19/all_data.csv',
    'vast': 'VAST/all_data.csv',
}

def read_csv(path):
    all_datas = []
    with open(path, 'r', encoding='utf-8') as read_f:
        reader = csv.DictReader(read_f)
        for row in reader:
            all_datas.append(row)
    return all_datas

field_names = ['Tweet', 'Target', 'Stance', 'discourse_knowledge']
def write_csv(path, datas, mode):
    if not os.path.isfile(path):
        with open(path, 'w', newline='', encoding='utf-8') as write_f:
            writer = csv.DictWriter(write_f, fieldnames=field_names)
            writer.writeheader()

    with open(path, mode, newline='', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        for data in datas:
            writer.writerow(data)

def read_json(path):
    with open(path, 'r') as read_f:
        data = json.loads(read_f.read())
    return data

def write_json(path, data):
    with open(path, 'w') as write_f:
        json.dump(data, write_f)

def is_valid_json(s):
    try:
        json.loads(s)['Restated Sentence']
        return True
    except json.JSONDecodeError:
        return False

def get_response(system_prompts, prompts, error_try=0):
    payload = {
        "model": "gpt-3.5-turbo-0301",
        "messages": [
            {"role": "system", "content": system_prompts},
            {"role": "user", "content": prompts}
        ]
    }
    try:
        response = ''
        escape_response = ''
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
        response = response['choices'][0]['message']['content'].strip()
        if error_try > 4:
            return None
        if is_valid_json(response):
            response = json.loads(response)['Restated Sentence']
        else:
            pattern = r'(?<!\{)(?<!: )(?<!\\)"(?!:)(?!,)(?!\})'  # 解析GPT回复中的双引号
            escape_response = response.replace('\":\"', '\": \"')
            escape_response = re.sub('{[ \n]*\"', '{\"', escape_response)
            escape_response = re.sub('\"[ \n]*}', '\"}', escape_response)
            escape_response = re.sub(pattern, '\\"', escape_response)
            response = json.loads(escape_response)['Restated Sentence']
            if response.count('\"') % 2 != 0:
                response = '\"' + response
        return response
    except Exception as e:
        logger.error(f'Get_response Error: {e}')
        logger.error(f'The trigger text is {response}')
        logger.error(f'The trigger escape text is {escape_response}')
        time.sleep(0.5)
        return get_response(system_prompts, prompts, error_try+1)


if __name__ == '__main__':
    for dataset_name in dataset_names:
        logging.info(f"Now is process dataset: {dataset_name}...")
        process_data_path = dataset_file_paths[dataset_name]
        logging.info(f'Processing {process_data_path}...')
        write_path = write_file_dir + '/' + process_data_path
        if not os.path.exists(os.path.dirname(write_path)):
            os.makedirs(os.path.dirname(write_path))
        all_datas = read_csv(f'{read_data_dir}/{process_data_path}')
        if os.path.exists(write_path):
            already_write_datas_num = len(read_csv(write_path))
        else:
            already_write_datas_num = 0
        for idx in tqdm(range(already_write_datas_num, len(all_datas))):
            try:
                sentence = all_datas[idx]['Tweet']
                target = all_datas[idx]['Target']
                system_prompt = ''
                prompt = discoures_knowledge_prompt % (target, sentence)
                response = get_response(system_prompt, prompt)
                copy_data = deepcopy(all_datas[idx])
                copy_data['discourse_knowledge'] = response
                write_csv(write_path, [copy_data], 'a')
            except KeyboardInterrupt as e:
                logging.info(f'KeyboardInterrupt {dataset_name}, {process_data_path}: {idx}/{len(all_datas)}')
                exit(0)
