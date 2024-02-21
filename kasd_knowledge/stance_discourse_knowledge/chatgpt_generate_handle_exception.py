import os
import csv
import json
import time
import logging
from tqdm import tqdm
from copy import deepcopy
import openai
from datetime import datetime

openai.api_key = '***'

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

write_file_dir = 'dataset/discourse_knowledge'
fh = logging.FileHandler(f'{write_file_dir}/log/' + 'openai_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +'.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

dataset_names = [
    'semeval16'
    'p-stance',
    'covid-19',
    'vast',
]

read_data_dir = 'dataset/processed_dataset_all'
dataset_file_paths = {
    'p-stance': [
        f'{read_data_dir}/P-Stance/all_data.csv',
    ],
    'vast': [
        f'{read_data_dir}/VAST/all_data.csv',
    ],
    'covid-19': [
        f'{read_data_dir}/COVID-19/all_data.csv',
    ],
    'semeval16': [
        f'{read_data_dir}/Semeval16/all_data.csv',
    ]
}
data_pointer_file_path = f'{write_file_dir}/data_pointer.json'

field_names = ['tweet_id', 'tweet_text', 'target', 'label', 'discourse_knowledge']

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

def get_response(system_prompt, prompts):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0301", 
            messages=[
                {'role': 'system', 'content': system_prompt},
                {"role": "user", "content": prompts}
            ]
        )
        response = response['choices'][0]['message']['content'].strip('\n')
    except Exception as e:
        logger.error(f'Error: {e}')
        time.sleep(25)
        return get_response(system_prompt, prompts)
    return response

if __name__ == '__main__':
    if not os.path.isfile(data_pointer_file_path):
        data_pointer = {}
        for dataset_name in dataset_names:
            data_pointer[dataset_name] = {}
            for process_data_path in dataset_file_paths[dataset_name]:
                data_pointer[dataset_name][process_data_path] = 0
        write_json(data_pointer_file_path, data_pointer)

    data_pointer = read_json(data_pointer_file_path)
    for dataset_name in dataset_names:
        logging.info(f"Now is process dataset: {dataset_name}...")
        if dataset_name not in data_pointer:
            data_pointer[dataset_name] = {}
        for process_data_path in dataset_file_paths[dataset_name]:
            logging.info(f'Processing {process_data_path}...')
            if process_data_path not in data_pointer[dataset_name]:
                data_pointer[dataset_name][process_data_path] = 0
            write_path = write_file_dir + '/' + process_data_path.split('/')[-2] + '/' + process_data_path.split('/')[-1]
            all_datas = read_csv(process_data_path)
            for idx in tqdm(range(data_pointer[dataset_name][process_data_path], len(all_datas))):
                try:
                    write_data = []
                    sentence = all_datas[idx]['tweet_text']
                    target = all_datas[idx]['target']
                    system_prompt = ''

                    prompt = f"""
                    The following is a text from Twitter about the topic "{target}". Please expand the abbreviations, slang and hash tags into complete sentences to restate the text. Please give your answer in json format and do not output anything unrelated to the task.
                    "{sentence}"
                    {{
                        "Restated Sentence" :
                    }}
                    """
                    response = get_response(system_prompt, prompt)
                    copy_data = deepcopy(all_datas[idx])
                    copy_data['discourse_knowledge'] = response
                    write_data.append(copy_data)
                    write_csv(write_path, write_data, 'a')
                except KeyboardInterrupt as e:
                    logging.info(f'KeyboardInterrupt {dataset_name}, {process_data_path}: {idx}/{len(all_datas)}')
                    data_pointer[dataset_name][process_data_path] = idx
                    write_json(data_pointer_file_path, data_pointer)
                    exit(0)
            data_pointer[dataset_name][process_data_path] = idx
            write_json(data_pointer_file_path, data_pointer)
