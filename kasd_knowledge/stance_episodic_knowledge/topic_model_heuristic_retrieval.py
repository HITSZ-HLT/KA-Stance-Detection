import csv
import random
random.seed(667)
from copy import deepcopy
import json
import logging
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sys
sys.path.append('./')
from utils.process_social_media_texts import split_hash_tag, clean_text

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

dataset_names = [
    'semeval16'
    'p-stance',
    'covid-19',
    'vast'
]

knowledge_paths = {
    'semeval16': 'dataset/topic_knowledge/Semeval16_topic_knowledge.json',
    'p-stance': 'dataset/topic_knowledge/P_stance_topic_knowledge.json',
    'covid-19': 'dataset/topic_knowledge/Covid_19_knowledge.json',
    'vast': 'dataset/topic_knowledge/VAST_topic_knowledge.json'
}

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

topic_map = {
    'p-stance': {
        'Donald Trump': 'Trump',
        'Joe Biden': 'Biden',
        'Bernie Sanders': 'Sanders'
    },
    'covid-19': {
        'face masks': 'Wearing a Face Mask',
        'fauci': 'Anthony S. Fauci, M.D.',
        'school closures': 'Keeping Schools Closed',
        'stay at home orders': 'Stay at Home Orders'
    },
    'vast': None,
    'semeval16': {
        'Atheism': 'Atheism',
        'Climate Change is a Real Concern': 'Climate Change is Concern',
        'Donald Trump': 'Donald Trump',
        'Feminist Movement': 'Feminist Movement',
        'Hillary Clinton': 'Hillary Clinton',
        'Legalization of Abortion': 'Legalization of Abortion'
    }
}

n_topics = {
    'p-stance': {
        'Trump': 17,
        'Biden': 17,
        'Sanders': 14,
    },
    'covid-19': {
        'Anthony S. Fauci, M.D.': 9,
        'Keeping Schools Closed': 11,
        'Stay at Home Orders': 8,
        'Wearing a Face Mask': 9,
    },
    'semeval16': {
        'Atheism': 13,
        'Climate Change is Concern': 17,
        'Donald Trump': 14,
        'Feminist Movement': 13,
        'Hillary Clinton': 17,
        'Legalization of Abortion': 15,
    }
}

topic_stop_words = {
    'Trump': ['donald', 'trump', 'president', 'united', 'state'],
    'Biden': ['joe', 'biden', 'president', 'united', 'state'],
    'Sanders': ['bernie', 'sanders', 'sander', 'president', 'united', 'state'],
    'Anthony S. Fauci, M.D.': ['fauci', 'covid19', 'dr', 'coronavirus', 'people', 'pandemic', 'md'],
    'Keeping Schools Closed': ['school', 'covid19', 'coronavirus', 'close', 'pandemic'],
    'Stay at Home Orders': ['covid19', 'coronavirus', 'order', 'people', 'pandemic', 'mask'],
    'Wearing a Face Mask': ['wear', 'people', 'covid19', 'coronavirus', 'virus', 'pandemic'],
    'Atheism': [],
    'Climate Change is Concern': [],
    'Donald Trump': [],
    'Feminist Movement': [],
    'Hillary Clinton': [],
    'Legalization of Abortion': [],
}

n_features = 7000
n_top_words = 50

write_file_dir = 'dataset/episodic_knowledge'

class TF_IDF_Model(object):
    def __init__(self, documents_list):
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 存储每个文本中每个词的词频
        self.tf = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(self.documents_number / (value + 1))

    def get_topic_score(self, index, topic_query, topic_query_token_scores):
        score = 0.0
        for idx, q in enumerate(topic_query):
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q] * topic_query_token_scores[idx] / len(topic_query)
        return score

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q] / len(query)
        return score

    def get_documents_score(self, query, topic_query, topic_query_token_scores, topic_query_weight):
        score = []
        topic_query_token_scores = np.array(topic_query_token_scores)
        topic_query_token_scores = topic_query_token_scores / topic_query_token_scores.mean()
        for topic_idx in range(len(topic_query)):
            score_list = []
            for i in range(self.documents_number):
                score_list.append(topic_query_weight[topic_idx] * self.get_topic_score(i, topic_query[topic_idx], topic_query_token_scores[topic_idx]))
            score.append(score_list)
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        score.append(score_list)
        score = np.array(score)

        return np.sum(score, axis=0)
    
def remove_stop_words(text, target_name, stop_word_dict=None):
    text_tokens = text.split(' ')
    if target_name in stop_word_dict:
        text_tokens = [tokens for tokens in text_tokens if tokens not in target_name.lower().split(' ') and tokens not in stop_word_dict[target_name]]
    else:
        text_tokens = [tokens for tokens in text_tokens if tokens not in target_name.lower().split(' ')]
    return ' '.join(text_tokens)

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

def read_knowledge(knowledge_path):
    orgin_docs = {}
    docs = {}
    docs_title = {}
    with open(knowledge_path, 'r', encoding='utf-8') as f_read:
        knowledge_datas = json.loads(f_read.read())
        for knowledge_data in tqdm(knowledge_datas):
            topic_name = knowledge_data['topic']
            topic_datas = knowledge_data['related_topics']
            for topic_data in topic_datas:
                knowledge_title = topic_data['title']
                spilt_knowledge_datas = topic_data['split_content']
                for spilt_knowledge_data in spilt_knowledge_datas:
                    if topic_name in docs_title and [knowledge_title] + spilt_knowledge_data[0] in docs_title[topic_name]:
                        continue
                    if 'External links' in spilt_knowledge_data[0] or 'See also' in spilt_knowledge_data[0] or 'References' in spilt_knowledge_data[0] or 'Further reading' in spilt_knowledge_data[0]:
                        continue
                    if topic_name not in docs_title:
                        orgin_docs[topic_name] = [spilt_knowledge_data[1]]
                        docs[topic_name] = [clean_text(spilt_knowledge_data[1]).split(' ')]
                        docs_title[topic_name] = [[knowledge_title] + spilt_knowledge_data[0]]
                    else:
                        orgin_docs[topic_name].append(spilt_knowledge_data[1])
                        docs[topic_name].append(clean_text(spilt_knowledge_data[1]).split(' '))
                        docs_title[topic_name].append([knowledge_title] + spilt_knowledge_data[0])
    return orgin_docs, docs, docs_title

KNOWLEDGE_TOPK = 10
def main():
    for dataset_name in dataset_names:
        logging.info(f"Now is process dataset: {dataset_name}...")
        logging.info(f'Load knowledge...')
        orgin_docs, docs, docs_title = read_knowledge(knowledge_paths[dataset_name])
        ori_train_datas = read_csv(dataset_file_paths[dataset_name][0])
        ori_valid_datas = read_csv(dataset_file_paths[dataset_name][1])
        ori_test_datas = read_csv(dataset_file_paths[dataset_name][2])
        train_write_datas = []
        valid_write_datas = []
        test_write_datas = []
        for topic_name, doc in docs.items():
            tf_idf_model = TF_IDF_Model(doc)
            corpus = []
            train_corpus = []
            valid_corpus = []
            test_corpus = []
            train_datas = []
            valid_datas = []
            test_datas = []
            for data in ori_train_datas:
                if topic_map[dataset_name][data['target']] == topic_name:
                    cleaned_text = clean_text(split_hash_tag(data['tweet_text']))
                    train_corpus.append(remove_stop_words(cleaned_text, topic_name, topic_stop_words))
                    train_datas.append(data)
            for data in ori_valid_datas:
                if topic_map[dataset_name][data['target']] == topic_name:
                    cleaned_text = clean_text(split_hash_tag(data['tweet_text']))
                    valid_corpus.append(remove_stop_words(cleaned_text, topic_name, topic_stop_words))
                    valid_datas.append(data)
            for data in ori_test_datas:
                if topic_map[dataset_name][data['target']] == topic_name:
                    cleaned_text = clean_text(split_hash_tag(data['tweet_text']))
                    test_corpus.append(remove_stop_words(cleaned_text, topic_name, topic_stop_words))
                    test_datas.append(data)
            corpus = train_corpus + valid_corpus + test_corpus

            tf_vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=2,
                max_features=n_features,
                stop_words='english')
            tf = tf_vectorizer.fit_transform(corpus)

            lda = LatentDirichletAllocation(
                n_components=n_topics[dataset_name][topic_name],
                max_iter=30,
                learning_method='batch',
                learning_offset=50.,
                random_state=0)
            lda.fit(tf)
            tf_feature_names = tf_vectorizer.get_feature_names()
            topic_tokens = []
            topic_token_scores = []
            for topic in lda.components_:
                topic_tokens.append([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
                norm_topic_score = (topic - np.min(topic)) / (np.max(topic) - np.min(topic))
                topic_token_scores.append(norm_topic_score[topic.argsort()[:-n_top_words - 1:-1]])
            
            data_topic = lda.transform(tf)
            assert data_topic.shape[0] == len(train_corpus) + len(valid_corpus) + len(test_corpus)
            
            for idx in range(0, len(train_corpus)):
                data_topic_idxs = data_topic[idx].argsort() #  从小到大
                data_topic_scores = data_topic[idx][data_topic[idx].argsort()]
                data_topic_tokens = []
                data_topic_token_scores = []
                for i in range(n_topics[dataset_name][topic_name]):
                    data_topic_tokens.append(topic_tokens[data_topic_idxs[i]])
                    data_topic_token_scores.append(topic_token_scores[data_topic_idxs[i]])
                tf_idf_score = tf_idf_model.get_documents_score(query=corpus[idx].split(' '), topic_query=data_topic_tokens, topic_query_token_scores=data_topic_token_scores, topic_query_weight=data_topic_scores)
                knowledge_idx = tf_idf_score.argsort()[-KNOWLEDGE_TOPK:]
                knowledge_sim_scores = tf_idf_score[tf_idf_score.argsort()[-KNOWLEDGE_TOPK:]]
                for i in range(KNOWLEDGE_TOPK):
                    data = deepcopy(train_datas[idx])
                    data['knowledge_title'] = docs_title[topic_map[dataset_name][data['target']]][knowledge_idx[i]]
                    data['knowledge'] = orgin_docs[topic_map[dataset_name][data['target']]][knowledge_idx[i]]
                    data['knowledge_sim_rank'] = KNOWLEDGE_TOPK-i
                    data['knowledge_sim_score'] = knowledge_sim_scores[i]
                    train_write_datas.append(data)
            for idx in range(len(train_corpus), len(train_corpus)+len(valid_corpus)):
                data_topic_idxs = data_topic[idx].argsort() #  从小到大
                data_topic_scores = data_topic[idx][data_topic[idx].argsort()]
                data_topic_tokens = []
                data_topic_token_scores = []
                for i in range(n_topics[dataset_name][topic_name]):
                    data_topic_tokens.append(topic_tokens[data_topic_idxs[i]])
                    data_topic_token_scores.append(topic_token_scores[data_topic_idxs[i]])
                tf_idf_score = tf_idf_model.get_documents_score(query=corpus[idx].split(' '), topic_query=data_topic_tokens, topic_query_token_scores=data_topic_token_scores, topic_query_weight=data_topic_scores)
                knowledge_idx = tf_idf_score.argsort()[-KNOWLEDGE_TOPK:]
                knowledge_sim_scores = tf_idf_score[tf_idf_score.argsort()[-KNOWLEDGE_TOPK:]]
                for i in range(KNOWLEDGE_TOPK):
                    data = deepcopy(valid_datas[idx-len(train_corpus)])
                    data['knowledge_title'] = docs_title[topic_map[dataset_name][data['target']]][knowledge_idx[i]]
                    data['knowledge'] = orgin_docs[topic_map[dataset_name][data['target']]][knowledge_idx[i]]
                    data['knowledge_sim_rank'] = KNOWLEDGE_TOPK-i
                    data['knowledge_sim_score'] = knowledge_sim_scores[i]
                    valid_write_datas.append(data)
            for idx in range(len(train_corpus)+len(valid_corpus), len(train_corpus)+len(valid_corpus)+len(test_corpus)):
                data_topic_idxs = data_topic[idx].argsort() #  从小到大
                data_topic_scores = data_topic[idx][data_topic[idx].argsort()]
                data_topic_tokens = []
                data_topic_token_scores = []
                for i in range(n_topics[dataset_name][topic_name]):
                    data_topic_tokens.append(topic_tokens[data_topic_idxs[i]])
                    data_topic_token_scores.append(topic_token_scores[data_topic_idxs[i]])
                tf_idf_score = tf_idf_model.get_documents_score(query=corpus[idx].split(' '), topic_query=data_topic_tokens, topic_query_token_scores=data_topic_token_scores, topic_query_weight=data_topic_scores)
                knowledge_idx = tf_idf_score.argsort()[-KNOWLEDGE_TOPK:]
                knowledge_sim_scores = tf_idf_score[tf_idf_score.argsort()[-KNOWLEDGE_TOPK:]]
                for i in range(KNOWLEDGE_TOPK):
                    data = deepcopy(test_datas[idx-len(train_corpus)-len(valid_corpus)])
                    data['knowledge_title'] = docs_title[topic_map[dataset_name][data['target']]][knowledge_idx[i]]
                    data['knowledge'] = orgin_docs[topic_map[dataset_name][data['target']]][knowledge_idx[i]]
                    data['knowledge_sim_rank'] = KNOWLEDGE_TOPK-i
                    data['knowledge_sim_score'] = knowledge_sim_scores[i]
                    test_write_datas.append(data)
        write_fields = ['tweet_id', 'tweet_text', 'target', 'label', 'knowledge_title', 'knowledge', 'knowledge_sim_rank', 'knowledge_sim_score']
        train_write_path = write_file_dir + '/' + dataset_file_paths[dataset_name][0].split('/')[-2] + '/' + dataset_file_paths[dataset_name][0].split('/')[-1]
        valid_write_path = write_file_dir + '/' + dataset_file_paths[dataset_name][1].split('/')[-2] + '/' + dataset_file_paths[dataset_name][1].split('/')[-1]
        test_write_path = write_file_dir + '/' + dataset_file_paths[dataset_name][2].split('/')[-2] + '/' + dataset_file_paths[dataset_name][2].split('/')[-1]
        write_csv(train_write_path, train_write_datas, write_fields)
        write_csv(valid_write_path, valid_write_datas, write_fields)
        write_csv(test_write_path, test_write_datas, write_fields)


def test_topic_model_ppl():
    for dataset_name in dataset_names:
        logging.info(f"Now is process dataset: {dataset_name}...")
        logging.info(f'Load knowledge...')
        orgin_docs, docs, docs_title = read_knowledge(knowledge_paths[dataset_name])
        ori_train_datas = read_csv(dataset_file_paths[dataset_name][0])
        ori_valid_datas = read_csv(dataset_file_paths[dataset_name][1])
        ori_test_datas = read_csv(dataset_file_paths[dataset_name][2])
        for topic_name, doc in docs.items():
            corpus = []
            train_corpus = []
            valid_corpus = []
            test_corpus = []
            for data in ori_train_datas:
                if topic_map[dataset_name][data['target']] == topic_name:
                    cleaned_text = clean_text(split_hash_tag(data['tweet_text']))
                    train_corpus.append(remove_stop_words(cleaned_text, topic_name, topic_stop_words))
            for data in ori_valid_datas:
                if topic_map[dataset_name][data['target']] == topic_name:
                    cleaned_text = clean_text(split_hash_tag(data['tweet_text']))
                    valid_corpus.append(remove_stop_words(cleaned_text, topic_name, topic_stop_words))
            for data in ori_test_datas:
                if topic_map[dataset_name][data['target']] == topic_name:
                    cleaned_text = clean_text(split_hash_tag(data['tweet_text']))
                    test_corpus.append(remove_stop_words(cleaned_text, topic_name, topic_stop_words))
            corpus = train_corpus + valid_corpus + test_corpus

            output_str = '\n' * 3 + f'topic_name: {topic_name}\n'
            pre_ppl_score = 0
            for n_topic_num in range(4, 20):
                tf_vectorizer = CountVectorizer(
                    max_df=0.95,
                    min_df=2,
                    max_features=n_features,
                    stop_words='english')
                tf = tf_vectorizer.fit_transform(corpus)

                lda = LatentDirichletAllocation(
                    n_components=n_topic_num,
                    max_iter=30,
                    learning_method='batch',
                    learning_offset=50.,
                    random_state=0)
                lda.fit(tf)
                tf_feature_names = tf_vectorizer.get_feature_names()
                topic_tokens = []
                topic_token_scores = []
                for topic in lda.components_:
                    topic_tokens.append([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
                    norm_topic_score = (topic - np.min(topic)) / (np.max(topic) - np.min(topic))
                    topic_token_scores.append(norm_topic_score[topic.argsort()[:-n_top_words - 1:-1]])
                
                ppl_score = lda.perplexity(tf)
                output_str += f'n_topic_num: {n_topic_num}, ppl_score: {ppl_score}, ppl_diff: {ppl_score-pre_ppl_score}\n'
                pre_ppl_score = ppl_score
                for tokens in topic_tokens:
                    output_str += ' '.join(tokens) + '\n'
                output_str += '==================================================\n'
            with open(f'Retrieval/{dataset_name}_lda_results.log', 'a') as f:
                f.write(output_str)
        with open(f'Retrieval/{dataset_name}_lda_results.log', 'a') as f:
                f.write('*' * 50)


if __name__ == '__main__':
    main()
    # test_topic_model_ppl()
