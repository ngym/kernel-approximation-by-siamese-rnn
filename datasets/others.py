# Lambdas to calculate labels from the sample names
GET_LABEL = dict.fromkeys(["6DMG", "6DMGupperChar", "upperChar"], lambda fn: fn.split('/')[-1].split('_')[1])
GET_LABEL["UCIcharacter"] = lambda str: str[0]
GET_LABEL["UCIauslan"] = lambda fn: fn.split('/')[-1].split('-')[0]


def get_label(dataset_type, sample_name):
    return GET_LABEL[dataset_type](sample_name)


def is_valid_dataset_type(dataset_type):
    return dataset_type in GET_LABEL