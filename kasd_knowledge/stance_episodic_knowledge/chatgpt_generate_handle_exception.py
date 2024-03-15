import re
import os
import csv
csv.field_size_limit(500 * 1024 * 1024)
import json
import time
import logging
from tqdm import tqdm
from copy import deepcopy
import requests
from datetime import datetime
import sys
sys.path.append('./')
from kasd_knowledge.prompts import episodic_knowledge_prompt

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

write_file_dir = 'datasets/episodic_knowledge'
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

read_data_dir = 'datasets/retrieved_knowledge'
dataset_file_paths = {
    'sem16': 'Semeval16/all_data.csv',
    'p_stance': 'P-Stance/all_data.csv',
    'covid_19': 'COVID-19/all_data.csv',
    'vast': 'VAST/all_data.csv',
}

field_names = ['Tweet', 'Target', 'Stance', 'knowledge_title', 'knowledge', 'knowledge_sim_rank', 'knowledge_sim_score', 'episodic_knowledge']

def read_csv(path):
    all_datas = []
    with open(path, 'r', encoding='utf-8') as read_f:
        reader = csv.DictReader(read_f)
        for row in reader:
            all_datas.append(row)
    return all_datas

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
        json.loads(s)['Output']
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
        if response.lower() == 'none' or response.lower() == 'none.':
            return None
        if is_valid_json(response):
            response = json.loads(response)['Output']
        else:
            pattern = r'(?<!\{)(?<!: )(?<!\\)"(?!:)(?!,)(?!\})'  # 解析GPT回复中的双引号
            escape_response = response.replace('\":\"', '\": \"')
            escape_response = re.sub('{[ \n]*\"', '{\"', escape_response)
            escape_response = re.sub('\"[ \n]*}', '\"}', escape_response)
            escape_response = re.sub(pattern, '\\"', escape_response)
            response = json.loads(escape_response)['Output']
            if response.count('\"') % 2 != 0:
                response = '\"' + response
        if response.lower() == 'none' or response.lower() == 'none.':
            return None
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
                episodic_knowledge = ''
                for knowledge_idx in range(10):
                    if eval(all_datas[idx]['knowledge_sim_score'])[knowledge_idx] < 0.02:
                        continue
                    sentence = all_datas[idx]['Tweet']
                    target = all_datas[idx]['Target']
                    document = eval(all_datas[idx]['knowledge'])[knowledge_idx]
                    system_prompt = ''
                    prompt = episodic_knowledge_prompt % (sentence, target, document)
                    response = get_response(system_prompt, prompt)
                    if response is None:
                        continue
                    episodic_knowledge += response + '\n'
                copy_data = deepcopy(all_datas[idx])
                copy_data['episodic_knowledge'] = episodic_knowledge
                write_csv(write_path, [copy_data], 'a')
            except KeyboardInterrupt as e:
                logging.info(f'KeyboardInterrupt {dataset_name}, {process_data_path}: {idx}/{len(all_datas)}')
                exit(0)