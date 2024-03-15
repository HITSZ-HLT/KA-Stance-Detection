class Covid19Config():
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    idx2label = {0: 'FAVOR', 1: 'AGAINST', 2: 'NONE'}
    label_tokens = ['favor', 'against', 'none']
    in_target_target_names = ['Stay at Home Orders', 'Keeping Schools Closed', 'Anthony S. Fauci, M.D.', 'Wearing a Face Mask']
    zero_shot_target_names = ['Stay at Home Orders', 'Keeping Schools Closed', 'Anthony S. Fauci, M.D.', 'Wearing a Face Mask']
    short_target_names = {
        'Anthony S. Fauci, M.D.': 'Fauci',
        'Stay at Home Orders': 'Home',
        'Wearing a Face Mask': 'Mask',
        'Keeping Schools Closed': 'School',
    }
    topic_text = {
        'Stay at Home Orders': 'stance on Stay at Home Orders',
        'Keeping Schools Closed': 'stance on Keeping Schools Closed',
        'Anthony S. Fauci, M.D.': 'stance on Anthony S. Fauci, M.D.',
        'Wearing a Face Mask': 'stance on Wearing a Face Mask',
    }

    data_dir = 'datasets/COVID-19'
    in_target_data_dir = f'{data_dir}/in-target'
    zero_shot_data_dir = f'{data_dir}/zero-shot'

    kasd_data_dir = 'datasets/kasd_dataset/COVID-19'
    in_target_kasd_data_dir = f'{kasd_data_dir}/in-target'
    zero_shot_kasd_data_dir = f'{kasd_data_dir}/zero-shot'