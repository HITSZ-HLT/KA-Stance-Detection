import os
import logging
import csv
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('./')
from utils.data_configs.vast_config import VASTConfig
from utils.process_social_media_texts import split_hash_tag, clean_text

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

class VASTDataset(Dataset):
    def __init__(self, tokenizer, target_name=None, text_with_target=True, if_split_hash_tag=True, in_target=None, zero_shot=None, train_data=None, valid_data=None, test_data=None, debug_mode=False):
        super(VASTDataset, self).__init__()
        assert train_data or valid_data or test_data
        assert in_target or zero_shot
        logging.info(f'Load VAST Zero shot Dataset')
        self.target_name = target_name
        self.text_with_target = text_with_target
        self.if_split_hash_tag = if_split_hash_tag

        if in_target:
            if train_data:
                self.data_path = f'{VASTConfig.in_target_data_dir}/{target_name}/train.csv'
            elif valid_data:
                self.data_path = f'{VASTConfig.in_target_data_dir}/{target_name}/valid.csv'
            elif test_data:
                self.data_path = f'{VASTConfig.in_target_data_dir}/{target_name}/test.csv'
        elif zero_shot:
            if train_data:
                self.data_path = f'{VASTConfig.zero_shot_data_dir}/{target_name}/train.csv'
            elif valid_data:
                self.data_path = f'{VASTConfig.zero_shot_data_dir}/{target_name}/valid.csv'
            elif test_data:
                self.data_path = f'{VASTConfig.zero_shot_data_dir}/{target_name}/test.csv'

        self.sentences, self.targets, self.labels = self.read_data(self.data_path, train_data, valid_data, test_data, debug_mode)
        self.encodings = tokenizer(self.sentences, self.targets, padding=True, truncation='only_first', return_tensors='pt')
        self.encodings['classification_label'] = torch.tensor(self.labels)
        if 'token_type_ids' not in self.encodings:
            self.encodings['token_type_ids'] = torch.zeros(self.encodings['input_ids'].shape, dtype=torch.int)

    def read_data(self, path, debug_mode=False):
        datas = []
        targets = []
        labels = []
        label_num = {}
        for label_name in VASTConfig.label2idx.keys():
            label_num[label_name] = 0
        all_datas = []

        with open(path, 'r', encoding='utf-8') as read_f:
            reader = csv.DictReader(read_f)
            for row in reader:
                all_datas.append(row)
        if debug_mode:
            all_datas = all_datas[:150]

        for data in all_datas:
            sentence = data['tweet_text']
            sentence = sentence.lstrip().rstrip()
            target = data['target']
            if self.if_split_hash_tag:
                sentence = split_hash_tag(sentence)
            else:
                sentence = sentence.lstrip().rstrip()
            if VASTConfig.apply_cleaning:
                sentence = clean_text(sentence)
            datas.append(sentence)
            targets.append(target)
            labels.append(VASTConfig.label2idx[data['label']])
            label_num[data['label']] += 1
        logging.info(f'loading data {len(datas)} from {path}')
        logging.info(f'label num ' + ' '.join([f'{k}: {v}' for k,v in label_num.items()]))
        return datas, targets, labels

    def __getitem__(self, item):
        return self.encodings['input_ids'][item], self.encodings['attention_mask'][item], self.encodings['token_type_ids'][item], self.encodings['classification_label'][item]

    def __len__(self):
        return len(self.encodings['input_ids'])

if __name__ == '__main__':
    from transformers import AutoTokenizer
    transformer_tokenizer_name = 'model_state/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer_name)
    tokenizer.model_max_length=tokenizer.max_model_input_sizes['/'.join(transformer_tokenizer_name.split('/')[1:])]

    VASTDataset(
        tokenizer=tokenizer,
        text_with_target=False,
        if_split_hash_tag=False,
        target_name=VASTConfig.target_names[0],
        in_target=True,
        train_data=True,
        debug_mode=True
    )