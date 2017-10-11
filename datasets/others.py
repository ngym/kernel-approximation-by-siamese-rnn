import os
from collections import OrderedDict

# Lambdas to calculate labels from the sample names
GET_LABEL = dict.fromkeys(["6DMG", "6DMGupperChar", "upperChar"], lambda fn: fn.split('/')[-1].split('_')[1])
GET_LABEL["UCIcharacter"] = lambda str: str[0]
GET_LABEL["UCIauslan"] = lambda fn: fn.split('/')[-1].split('-')[0]

GET_SAMPLE_NAME = dict.fromkeys(["6DMG", "6DMGupperChar", "upperChar"], lambda fn: fn.split('/')[-1].replace('.mat', ''))
GET_SAMPLE_NAME["UCIcharacter"] = lambda str: str
GET_SAMPLE_NAME["UCIauslan"] = lambda fn: os.path.join(*fn.split('/')[-2:])


def get_label(dataset_type, sample_name):
    return GET_LABEL[dataset_type](sample_name)


def is_valid_dataset_type(dataset_type):
    return dataset_type in GET_LABEL


def get_sample_name(dataset_type, sample_file):
    return GET_SAMPLE_NAME[dataset_type](sample_file)

def filter_samples(order_dict, sample_names):
    od = OrderedDict()
    for sn in sample_names:
        od[sn] = order_dict[sn]
    return od



