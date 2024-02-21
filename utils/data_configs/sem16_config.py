class Sem16Config():
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1]
    label2idx = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    idx2label = {0: 'FAVOR', 1: 'AGAINST', 2: 'NONE'}
    short_target_names = {
        'FM': 'Feminist Movement',
        'HC': 'Hillary Clinton',
        'LA': 'Legalization of Abortion'
    }

    target_names = ['Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion']
    topic_text = {
        'Feminist Movement': 'stance on Feminist Movement',
        'Hillary Clinton': 'stance on Hillary Clinton',
        'Legalization of Abortion': 'stance on Legalization of Abortion',
        'Atheism': 'stance on Atheism',
        'Climate Change is a Real Concern': 'stance on Climate Change is a Real Concern',
        'Donald Trump': 'stance on Donald Trump',
    }
    target_2_knowledge_name = {
        'Atheism': 'Atheism',
        'Climate Change is a Real Concern': 'Climate Change is Concern',
        'Donald Trump': 'Donald Trump',
        'Feminist Movement': 'Feminist Movement',
        'Hillary Clinton': 'Hillary Clinton',
        'Legalization of Abortion': 'Legalization of Abortion'
    }

    data_dir = 'dataset/processed_dataset/Semeval16'
    in_target_data_dir = f'{data_dir}/in_target'
    zero_shot_data_dir = f'{data_dir}/zero_shot'

    kasd_data_dir = 'dataset/kasd_dataset/Semeval16'
    in_target_kasd_data_dir = f'{kasd_data_dir}/in_target'
    zero_shot_kasd_data_dir = f'{kasd_data_dir}/zero_shot'