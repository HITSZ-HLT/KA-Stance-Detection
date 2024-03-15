# Introduction

This repository open-sources the code and part of datas used in our [paper](https://aclanthology.org/2023.emnlp-main.972/)「**Stance Detection on Social Media with Background Knowledge**」in EMNLP2023 main conference long paper.

<img src="figures\framework.svg" width = "60%" />

Please cite our paper and kindly give a star for this repository if you use our code or data.

# Requirements

Seeing in requirement.txt

You could using `pip install -r requirement.txt` to install the required packages.

# Usage

## Dataset

Download the [Sem16](https://alt.qcri.org/semeval2016/task6), [P-stance](https://github.com/chuchun8/PStance), [Covid-19](https://github.com/kglandt/stance-detection-in-covid-19-tweets) and [VAST](https://github.com/emilyallaway/zero-shot-stance) or other stance detection dataset, place them into `dataset/raw_dataset/<dataset name>`

Process the datasets into the following format:

```
# Each file is a csv file, containing at least the three keys 'tweet_text', 'target', 'label'
- datasets
  - <dataset name>
    - in-target
      - <target name>
        - train.csv
        - valid.csv
        - test.csv
      - <target name>
        - ...
    - zero-shot
      - <target name>
        - train.csv
        - valid.csv
        - test.csv
      - <target name>
        - ...
  - <dataset name>
    - ...
```

The way of how I process the datasets is shown in `datasets/preprocess_datasets.py`

Download our open-sourced knowledge from [Baidu Drive](https://pan.baidu.com/s/1Ou6GWUzzP2PkyO1zgy1KLA?pwd=6gwa), and unzip them into folder `datasets/topic_knowledge`

Download your needed model states into `model_state` or remove all `model_state/` dir prefix in all config files in `configs`.

## Obtaining Episodic Knowledge and Discourse Knowledge

```
sh scripts/kasd_knowledge.sh
```

## Stance Detection on Social Media with Background Knowledge

```
sh scripts\baseline\bert_based\train.sh
```

Take in-target stance detection on p-stance for example

```
>>> sh scripts\baseline\bert_based\train.sh
>>> input training dataset: [p_stance, sem16, covid_19, vast]: p_stance
>>> input train dataset mode: [in_target, zero_shot]: in_target
>>> input model name: [roberta_base, roberta_large, bertweet_base, bertweet_large, ct_bert_large]: roberta_base
>>> input model framework: [base, kasd]: kasd
>>> input running mode: [sweep, wandb, normal]: normal
>>> input training cuda idx: Your Cuda index
```

# Citation

The BibTex of the citation is as follows:

```bibtex
@inproceedings{li-etal-2023-stance,
    title = "Stance Detection on Social Media with Background Knowledge",
    author = "Li, Ang  and
      Liang, Bin  and
      Zhao, Jingqian  and
      Zhang, Bowen  and
      Yang, Min  and
      Xu, Ruifeng",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.972",
    pages = "15703--15717",
    abstract = "Identifying users{'} stances regarding specific targets/topics is a significant route to learning public opinion from social media platforms. Most existing studies of stance detection strive to learn stance information about specific targets from the context, in order to determine the user{'}s stance on the target. However, in real-world scenarios, we usually have a certain understanding of a target when we express our stance on it. In this paper, we investigate stance detection from a novel perspective, where the background knowledge of the targets is taken into account for better stance detection. To be specific, we categorize background knowledge into two categories: episodic knowledge and discourse knowledge, and propose a novel Knowledge-Augmented Stance Detection (KASD) framework. For episodic knowledge, we devise a heuristic retrieval algorithm based on the topic to retrieve the Wikipedia documents relevant to the sample. Further, we construct a prompt for ChatGPT to filter the Wikipedia documents to derive episodic knowledge. For discourse knowledge, we construct a prompt for ChatGPT to paraphrase the hashtags, references, etc., in the sample, thereby injecting discourse knowledge into the sample. Experimental results on four benchmark datasets demonstrate that our KASD achieves state-of-the-art performance in in-target and zero-shot stance detection.",
}
```

# Poster

A poster of our work is as follows:

<img src="figures\emnlp-poster-official.png" width = "90%" />