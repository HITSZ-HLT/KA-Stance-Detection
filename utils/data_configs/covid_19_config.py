class Covid19Config():
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    idx2label = {0: 'FAVOR', 1: 'AGAINST', 2: 'NONE'}
    label_tokens = ['favor', 'against', 'none']
    target_names = ['stay at home orders', 'school closures', 'fauci', 'face masks']
    short_target_names = {
        'Fauci': 'fauci',
        'Home': 'stay at home orders',
        'Mask': 'face masks',
        'School': 'school closures',
    }
    topic_text = {
        'stay at home orders': 'stance on Stay at Home Orders',
        'school closures': 'stance on Keeping Schools Closed',
        'fauci': 'stance on Anthony S. Fauci, M.D',
        'face masks': 'stance on Wearing a Face Mask',
    }
    target_2_knowledge_name = {
        'stay at home orders': 'Stay at Home Orders', 
        'school closures': 'Keeping Schools Closed', 
        'fauci': 'Anthony S. Fauci, M.D.', 
        'face masks': 'Wearing a Face Mask',
    }

    data_dir = 'dataset/processed_dataset/COVID-19'
    in_target_data_dir = f'{data_dir}/in_target'
    zero_shot_data_dir = f'{data_dir}/zero_shot'

    kasd_data_dir = 'dataset/kasd_dataset/COVID-19'
    in_target_kasd_data_dir = f'{kasd_data_dir}/in_target'
    zero_shot_kasd_data_dir = f'{kasd_data_dir}/zero_shot'