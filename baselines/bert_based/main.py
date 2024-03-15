import sys
sys.path.append('./')
import copy
import os
import numpy as np
import logging
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils.tools import set_seed, read_yaml, AttributeDict
transformers.logging.set_verbosity_error()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug_mode', action='store_true')
parser.add_argument('--cuda_idx', type=int)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--model_name', type=str, default='bertweet_base')
parser.add_argument('--framework_name', type=str, default='base')
parser.add_argument('--in_target', action='store_true')
parser.add_argument('--zero_shot', action='store_true')
parser.add_argument('--multi_target', action='store_true')
parser.add_argument('--sweep', action='store_true')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--normal', action='store_true')
GLOBAL_ARGS = parser.parse_args()

RUN_FRAMEWORK_NAME = 'baseline_model'
PROJECT_NAME = 'KASD'

if GLOBAL_ARGS.dataset_name == 'vast':
    assert GLOBAL_ARGS.zero_shot
if GLOBAL_ARGS.zero_shot:
    DATASET_NAME = GLOBAL_ARGS.dataset_name + '_zero_shot'
elif GLOBAL_ARGS.in_target:
    DATASET_NAME = GLOBAL_ARGS.dataset_name + '_in_target'

if GLOBAL_ARGS.framework_name == 'base':
    from data_config import data_configs as data_configs
    from data_config import datasets as datasets
elif GLOBAL_ARGS.framework_name == 'kasd':
    from data_config import data_configs as data_configs
    from data_config import kasd_datasets as datasets
else:
    raise ValueError('Invalid model')

# Set seeds for reproducibility
SEED = [42, 67, 2022, 31, 15]

# if to save the model states
SAVE_STATE = False
if GLOBAL_ARGS.debug_mode:
    print('=====================debug_mode=======================')
    SAVE_STATE = False
elif GLOBAL_ARGS.normal:
    pass
else:
    import wandb
    wandb.login()

FILE_PATH = __file__
OUTPUT_DIR = os.path.dirname(__file__) + '/outputs'
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
CUR_DIR = OUTPUT_DIR + f'/{DATASET_NAME}_{GLOBAL_ARGS.model_name}_{GLOBAL_ARGS.framework_name}_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

CUR_MODELS_DIR = CUR_DIR + '/models'
if not GLOBAL_ARGS.debug_mode:
    if not os.path.isdir(CUR_DIR):
        os.makedirs(CUR_DIR)
        os.makedirs(CUR_MODELS_DIR)

# load config
CONFIG_PATH = {
    'base': 'configs/bert_based_config.yaml',
    'kasd': 'configs/kasd_config.yaml',
}

raw_config = read_yaml(CONFIG_PATH[GLOBAL_ARGS.framework_name])

if GLOBAL_ARGS.sweep:
    SAVE_STATE = False

if GLOBAL_ARGS.model_name in raw_config['sweep_config']['model_config']:
    raw_config['sweep_config']['model_config'] = raw_config['sweep_config']['model_config'][GLOBAL_ARGS.model_name]
else:
    raise ValueError('Invalid model')
if GLOBAL_ARGS.model_name in raw_config['recommended_settings']['model_config']:
    raw_config['recommended_settings']['model_config'] = raw_config['recommended_settings']['model_config'][GLOBAL_ARGS.model_name]
else:
    raise ValueError('Invalid model')

from models import Bert_model as Model

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)
if not GLOBAL_ARGS.debug_mode:
    fhlr = logging.FileHandler(CUR_DIR + '/training_result.log', mode='w')
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)

LABELS = data_configs[GLOBAL_ARGS.dataset_name].test_stance
DEVICE = torch.device('cuda:' + str(GLOBAL_ARGS.cuda_idx))

def build_dataset(args, tokenizer, target_name=None):
    train_dataset_orig = datasets[GLOBAL_ARGS.dataset_name](
        tokenizer=tokenizer,
        target_name=target_name,
        text_with_target=args.text_with_target,
        if_split_hash_tag=args.if_split_hash_tag,
        in_target=GLOBAL_ARGS.in_target,
        zero_shot=GLOBAL_ARGS.zero_shot,
        train_data=True,
        debug_mode=GLOBAL_ARGS.debug_mode)
    valid_dataset_orig = datasets[GLOBAL_ARGS.dataset_name](
        tokenizer=tokenizer,
        target_name=target_name,
        text_with_target=args.text_with_target,
        if_split_hash_tag=args.if_split_hash_tag,
        in_target=GLOBAL_ARGS.in_target,
        zero_shot=GLOBAL_ARGS.zero_shot,
        valid_data=True,
        debug_mode=GLOBAL_ARGS.debug_mode)
    test_dataset_orig = datasets[GLOBAL_ARGS.dataset_name](
        tokenizer=tokenizer,
        target_name=target_name,
        text_with_target=args.text_with_target,
        if_split_hash_tag=args.if_split_hash_tag,
        in_target=GLOBAL_ARGS.in_target,
        zero_shot=GLOBAL_ARGS.zero_shot,
        test_data=True,
        debug_mode=GLOBAL_ARGS.debug_mode)

    train_data_loader = DataLoader(
        train_dataset_orig,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    valid_data_loader = DataLoader(
        valid_dataset_orig,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4)
    test_data_loader = DataLoader(
        test_dataset_orig,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4)
    return train_data_loader, valid_data_loader, test_data_loader


def train(train_data_loader, model, criterion, optimizer, scheduler):
    model.train()
    loss_mean = 0.0

    for train_data in train_data_loader:
        train_data = [data.to(DEVICE) for data in train_data]
        input_data = train_data[:-1]
        y = train_data[-1]

        optimizer.zero_grad()

        logits = model(input_data)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    return loss_mean / len(train_data_loader)

@torch.no_grad()
def evaluate(eval_data_loader, model, criterion):
    model.eval()

    loss_mean = 0.0

    true_labels = []
    predict_labels = []

    for eval_data in eval_data_loader:
        eval_data = [data.to(DEVICE) for data in eval_data]
        input_data = eval_data[:-1]
        y = eval_data[-1]

        logits = model(input_data)
        loss = criterion(logits, y)
        loss_mean += loss.item()

        true_labels.append(y.cpu())
        predict_labels.append(torch.argmax(logits, dim=-1).cpu())

    true_labels = torch.cat(true_labels, dim=0)
    predict_labels = torch.cat(predict_labels, dim=0)
    f1 = f1_score(true_labels, predict_labels, average='macro', labels=LABELS) * 100
    precision = precision_score(true_labels, predict_labels, average='macro', labels=LABELS) * 100
    recall = recall_score(true_labels, predict_labels, average='macro', labels=LABELS) * 100
    acc = accuracy_score(true_labels, predict_labels) * 100

    return loss_mean / len(eval_data_loader), acc, f1, precision, recall


def train_process(args, tokenizer, train_data_loader, valid_data_loader, test_data_loader, fold=0, target_name=None, save_state=True):
    logging.info('init models, optimizer, criterion...')
    model = Model(args).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    transformer_identifiers = ['shared.weight', 'embedding', 'encoder', 'decoder', 'pooler']
    linear_identifiers = ['linear', 'classifier']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in transformer_identifiers) and
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
        'lr': args.transformer_learning_rate,
        'weight_decay': args.weight_decay},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in transformer_identifiers) and
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
        'lr': args.transformer_learning_rate,
        'weight_decay': 0.0},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in linear_identifiers) and 
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
        'lr': args.linear_learning_rate,
        'weight_decay': args.weight_decay},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in linear_identifiers) and 
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
        'lr': args.linear_learning_rate,
        'weight_decay': 0.0}
    ]

    optimizer = optim.AdamW(grouped_model_parameters)
    total_steps = len(train_data_loader) * args.num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=total_steps*args.warmup_ratio,
                                                num_training_steps=total_steps)

    logging.info('Start training...')
    best_state = None
    best_valid_f1 = 0.0
    best_test_f1 = 0.0
    best_test_acc = 0.0
    best_test_precision = 0.0
    best_test_recall = 0.0
    temp_path = CUR_MODELS_DIR + \
        f'/{DATASET_NAME}_temp_model.pt'
    for ep in range(args.num_epochs):
        logging.info(f'epoch {ep+1} start train')
        train_loss = train(train_data_loader, model, criterion, optimizer, scheduler)
        logging.info(f'epoch {ep+1} start evaluate')
        valid_loss, valid_acc, valid_f1, valid_precision, valid_recall = evaluate(valid_data_loader, model, criterion)
        test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(test_data_loader, model, criterion)
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_test_f1 = test_f1
            best_test_acc = test_acc
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_path = CUR_MODELS_DIR + \
                f'/{DATASET_NAME}_{target_name}_fold_{fold}_{best_test_f1:.5f}_{datetime.now().strftime("%m-%d-%H-%M")}.pt'
            best_state = copy.deepcopy(model.state_dict())
            if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
                wandb.log({f'{fold}_epoch': ep+1, "best_valid_f1": best_valid_f1, "best_test_f1": best_test_f1})

            if ep > 3 and best_state != None and save_state:
                logging.info(f'saving best model acc {best_test_acc:.5f}, f1 {best_test_f1:.5f}, precision {best_test_precision:.5f}, recall {best_test_recall:.5f} in {temp_path}')
                torch.save(best_state, temp_path)

        if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
            wandb.log({f'{fold}_epoch': ep+1, "train_loss": train_loss, "valid_loss": valid_loss, "test_loss": test_loss, "valid_f1": valid_f1, "test_f1": test_f1})
        logging.info(f'epoch {ep+1} done! train_loss {train_loss:.5f}, valid_loss {valid_loss:.5f}, test_loss {test_loss:.5f}')
        logging.info(f'valid: acc {valid_acc:.5f}, f1 {valid_f1:.5f}, precision {valid_precision:.5f}, recall {valid_recall:.5f}')
        logging.info(f'test: acc {test_acc:.5f}, f1 {test_f1:.5f}, precision {test_precision:.5f}, recall {test_recall:.5f}, now best_f1 {best_test_f1:.5f}')

    if best_state != None and save_state:
        logging.info(f'saving best model acc {best_test_acc:.5f}, f1 {best_test_f1:.5f}, precision {best_test_precision:.5f}, recall {best_test_recall:.5f}, in {best_path}')
        torch.save(best_state, best_path)

    return best_test_acc, best_test_f1, best_test_precision, best_test_recall

def main(args=None):
    if GLOBAL_ARGS.zero_shot:
        train_type = 'zero_shot'
    else:
        train_type = 'in_target'
    if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
        run_name = f'{RUN_FRAMEWORK_NAME}_{train_type}_{GLOBAL_ARGS.dataset_name}_{GLOBAL_ARGS.model_name}_{GLOBAL_ARGS.framework_name}_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
        wandb.init(project=PROJECT_NAME, name=run_name, config=args)
        args = wandb.config
    set_seed()
    logging.info('Using cuda device gpu: ' + str(GLOBAL_ARGS.cuda_idx))
    if not GLOBAL_ARGS.debug_mode:
        logging.info('Saving into directory ' + CUR_DIR)
    logging.info(f'model name: {args.transformer_name}')

    output_str = ''
    final_score = 0
    for target_name in data_configs[GLOBAL_ARGS.dataset_name].in_target_target_names if train_type == 'in_target' else data_configs[GLOBAL_ARGS.dataset_name].zero_shot_target_names:
        acc = []
        f1 = []
        precision = []
        recall = []
        logging.info('preparing data...')
        tokenizer = AutoTokenizer.from_pretrained(args.transformer_tokenizer_name, model_max_length=args.max_tokenization_length)
        for fold in range(args.train_times):
            logging.info(f'all train times:{args.train_times}, now is fold:{fold+1}')
            logging.info('preparing data...')
            set_seed(SEED[fold])

            train_data_loader, valid_data_loader, test_data_loader = build_dataset(args, tokenizer, target_name)

            best_acc, best_f1, best_precision, best_recall = train_process(args, tokenizer, train_data_loader, valid_data_loader, test_data_loader, fold=fold, target_name=target_name, save_state=SAVE_STATE)
            acc.append(best_acc)
            f1.append(best_f1)
            precision.append(best_precision)
            recall.append(best_recall)

            logging.info(f'target "{target_name}" fold {fold+1} train finish, acc:{best_acc:.5f}, f1:{best_f1:.5f}, precision:{best_precision:.5f}, recall:{best_recall:.5f}')
        acc_avg = np.mean(acc)
        f1_avg = np.mean(f1)
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
        acc_std = np.std(acc, ddof=0)
        f1_std = np.std(f1, ddof=0)
        precision_std = np.std(precision, ddof=0)
        recall_std = np.std(recall, ddof=0)
        acc_sem = acc_std / np.sqrt(args.train_times)
        f1_sem = f1_std / np.sqrt(args.train_times)
        precision_sem = precision_std / np.sqrt(args.train_times)
        recall_sem = recall_std / np.sqrt(args.train_times)

        logging.info(f'target "{target_name}" train finish')
        result = {'target_name': data_configs[GLOBAL_ARGS.dataset_name].short_target_names[target_name], 'acc_avg': acc_avg, 'f1_avg': f1_avg, 'precision_avg': precision_avg, 'recall_avg': recall_avg, 'acc_std': acc_std, 'f1_std': f1_std, 'precision_std': precision_std, 'recall_std': recall_std, 'acc_sem': acc_sem, 'f1_sem': f1_sem, 'precision_sem': precision_sem, 'recall_sem': recall_sem}
        final_score += f1_avg
        for key in result.values():
            if type(key) is str:
                output_str += f'{key}\t'
            else:
                output_str += f'{key:.5f}\t'
        output_str += '\n'

    logging.info(f'all train finish')
    output_str = f'\ntarget_name\tacc_avg:\tf1_avg:\tprecision_avg:\trecall_avg:\tacc_std\tf1_std\tprecision_std\trecall_std\tacc_sem\tf1_sem\tprecision_sem\trecall_sem\n' + output_str
    logging.info(output_str)
    final_score = final_score / len(data_configs[GLOBAL_ARGS.dataset_name].in_target_target_names if train_type == 'in_target' else data_configs[GLOBAL_ARGS.dataset_name].zero_shot_target_names)
    if not GLOBAL_ARGS.debug_mode and not GLOBAL_ARGS.normal:
        wandb.log({'final_score': final_score})


if __name__ == '__main__':
    logging.info(f'pid: {os.getpid()}')
    if GLOBAL_ARGS.debug_mode or GLOBAL_ARGS.normal:
        raw_config['recommended_settings']['model_config']['label_size'] = len(data_configs[GLOBAL_ARGS.dataset_name].label2idx)
        if GLOBAL_ARGS.debug_mode:
            raw_config['recommended_settings']['train_config']['num_epochs'] = 5
        config = {}
        for value in raw_config['recommended_settings'].values():
            config.update(value)
        config = AttributeDict(config)
        main(config)
    elif GLOBAL_ARGS.sweep:
        raw_config['sweep_config']['model_config']['label_size']['value'] = len(data_configs[GLOBAL_ARGS.dataset_name].label2idx)
        sweep_config = {
            'method': 'random',
            'metric': {'goal': 'maximize', 'name': 'final_score'},
            'parameters': {}
        }
        for value in raw_config['sweep_config'].values():
            sweep_config['parameters'].update(value)
        sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
        wandb.agent(sweep_id, main, count=50)
    else:
        raw_config['recommended_settings']['model_config']['label_size'] = len(data_configs[GLOBAL_ARGS.dataset_name].label2idx)
        config = {}
        for value in raw_config['recommended_settings'].values():
            config.update(value)
        main(config)