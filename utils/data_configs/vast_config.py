class VASTConfig():
    is_multi_label = True
    apply_cleaning = False
    test_stance = [0, 1, 2]
    label2idx = {'pro': 0, 'con': 1, 'neutral': 2}
    idx2label = {0: 'pro', 1: 'con', 2: 'neutral'}
    zero_shot_target_names = ['zero-shot']
    short_target_names = {
        'zero-shot': 'zero-shot'
    }

    data_dir = 'datasets/VAST'
    zero_shot_data_dir = f'{data_dir}'

    kasd_data_dir = 'datasets/kasd_dataset/VAST'
    zero_shot_kasd_data_dir = f'{kasd_data_dir}'