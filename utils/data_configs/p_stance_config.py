class PStanceConfig():
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1]
    label2idx = {'FAVOR': 0, 'AGAINST': 1,}
    idx2label = {0: 'FAVOR', 1: 'AGAINST'}
    label_tokens = ['favor', 'against']
    in_target_target_names = ['Donald Trump', 'Joe Biden', 'Bernie Sanders']
    zero_shot_target_names = ['Donald Trump', 'Joe Biden', 'Bernie Sanders']
    short_target_names = {
        'Donald Trump': 'DT',
        'Joe Biden': 'JB',
        'Bernie Sanders': 'BS',
    }
    topic_text = {
        'Donald Trump': 'stance on Donald Trump',
        'Joe Biden': 'stance on Joe Biden',
        'Bernie Sanders': 'stance on Bernie Sanders'
    }

    data_dir = 'datasets/P-Stance'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'

    kasd_data_dir = 'datasets/kasd_dataset/P-Stance'
    in_target_kasd_data_dir = f'{kasd_data_dir}/in-target'
    zero_shot_kasd_data_dir = f'{kasd_data_dir}/zero-shot'