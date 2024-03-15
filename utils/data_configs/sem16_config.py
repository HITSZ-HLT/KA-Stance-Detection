class Sem16Config():
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1]
    label2idx = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    idx2label = {0: 'FAVOR', 1: 'AGAINST', 2: 'NONE'}
    in_target_target_names = ['Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion', 'Atheism', 'Climate Change is a Real Concern']
    zero_shot_target_names = ['Feminist Movement', 'Hillary Clinton', 'Legalization of Abortion', 'Atheism', 'Climate Change is a Real Concern', 'Donald Trump']
    short_target_names = {
        'Feminist Movement': 'FM',
        'Hillary Clinton': 'HC',
        'Legalization of Abortion': 'LA',
        'Atheism': 'A',
        'Climate Change is a Real Concern': 'CC',
        'Donald Trump': 'DT',
    }
    topic_text = {
        'Feminist Movement': 'stance on Feminist Movement',
        'Hillary Clinton': 'stance on Hillary Clinton',
        'Legalization of Abortion': 'stance on Legalization of Abortion',
        'Atheism': 'stance on Atheism',
        'Climate Change is a Real Concern': 'stance on Climate Change is a Real Concern',
        'Donald Trump': 'stance on Donald Trump',
    }

    data_dir = 'datasets/Semeval16'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'

    kasd_data_dir = 'datasets/kasd_dataset/Semeval16'
    in_target_kasd_data_dir = f'{kasd_data_dir}/in-target'
    zero_shot_kasd_data_dir = f'{kasd_data_dir}/zero-shot'