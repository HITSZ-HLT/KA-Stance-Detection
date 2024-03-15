import os
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
    'sem16',
    'p_stance',
    'covid_19',
    'vast'
]

knowledge_paths = {
    'sem16': 'datasets/topic_knowledge/Semeval16_topic_knowledge.json',
    'p_stance': 'datasets/topic_knowledge/P_stance_topic_knowledge.json',
    'covid_19': 'datasets/topic_knowledge/Covid_19_knowledge.json',
    'vast': 'datasets/topic_knowledge/VAST_topic_knowledge.json',
}

read_data_dir = 'datasets/processed_dataset_all'
dataset_file_paths = {
    'sem16': 'Semeval16/all_data.csv',
    'p_stance': 'P-Stance/all_data.csv',
    'vast': 'VAST/all_data.csv',
    'covid_19': 'COVID-19/all_data.csv',
}

topic_map = {
    'p_stance': {
        'Donald Trump': 'Trump',
        'Joe Biden': 'Biden',
        'Bernie Sanders': 'Sanders'
    },
    'covid_19': {
        'Wearing a Face Mask': 'Wearing a Face Mask',
        'Anthony S. Fauci, M.D.': 'Anthony S. Fauci, M.D.',
        'Keeping Schools Closed': 'Keeping Schools Closed',
        'Stay at Home Orders': 'Stay at Home Orders'
    },
    'vast': None,
    'sem16': {
        'Atheism': 'Atheism',
        'Climate Change is a Real Concern': 'Climate Change is Concern',
        'Donald Trump': 'Donald Trump',
        'Feminist Movement': 'Feminist Movement',
        'Hillary Clinton': 'Hillary Clinton',
        'Legalization of Abortion': 'Legalization of Abortion'
    }
}

n_topics = {
    'p_stance': {
        'Trump': 17,
        'Biden': 17,
        'Sanders': 14,
    },
    'covid_19': {
        'Anthony S. Fauci, M.D.': 9,
        'Keeping Schools Closed': 11,
        'Stay at Home Orders': 8,
        'Wearing a Face Mask': 9,
    },
    'sem16': {
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

write_file_dir = 'datasets/retrieved_knowledge'

class TF_IDF_Model(object):
    def __init__(self, documents_list):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.tf = []
        self.idf = {}
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
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
        ori_datas = read_csv(f'{read_data_dir}/{dataset_file_paths[dataset_name]}')
        write_datas = []
        for topic_name, doc in docs.items():
            tf_idf_model = TF_IDF_Model(doc)
            corpus = []
            datas = []
            for data in ori_datas:
                if topic_map[dataset_name][data['Target']] == topic_name:
                    cleaned_text = clean_text(split_hash_tag(data['Tweet']))
                    corpus.append(remove_stop_words(cleaned_text, topic_name, topic_stop_words))
                    datas.append(data)

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
            tf_feature_names = tf_vectorizer.get_feature_names_out()
            topic_tokens = []
            topic_token_scores = []
            for topic in lda.components_:
                topic_tokens.append([tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
                norm_topic_score = (topic - np.min(topic)) / (np.max(topic) - np.min(topic))
                topic_token_scores.append(norm_topic_score[topic.argsort()[:-n_top_words - 1:-1]])
            
            data_topic = lda.transform(tf)
            for idx in range(0, len(corpus)):
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
                data = deepcopy(datas[idx])
                data['knowledge_title'] = []
                data['knowledge'] = []
                data['knowledge_sim_rank'] = []
                data['knowledge_sim_score'] = []
                for i in range(KNOWLEDGE_TOPK):
                    data['knowledge_title'].append(docs_title[topic_map[dataset_name][data['Target']]][knowledge_idx[i]])
                    data['knowledge'].append(orgin_docs[topic_map[dataset_name][data['Target']]][knowledge_idx[i]])
                    data['knowledge_sim_rank'].append(KNOWLEDGE_TOPK-i)
                    data['knowledge_sim_score'].append(knowledge_sim_scores[i])
                write_datas.append(data)
        write_fields = ['Tweet', 'Target', 'Stance', 'knowledge_title', 'knowledge', 'knowledge_sim_rank', 'knowledge_sim_score']
        write_path = write_file_dir + '/' + dataset_file_paths[dataset_name]
        if not os.path.exists(os.path.dirname(write_path)):
            os.makedirs(os.path.dirname(write_path))
        write_csv(write_path, write_datas, write_fields)


if __name__ == '__main__':
    main()