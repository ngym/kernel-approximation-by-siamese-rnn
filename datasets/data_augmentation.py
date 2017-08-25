import sys, os, csv
from collections import OrderedDict
import glob

import numpy as np
from scipy import io
from scipy.stats import truncnorm

from datasets import others

from datasets.read_sequences import read_sequences
from utils import file_utils

def augment_data(seqs, key_to_str, length, rand_uniform=True, num_normaldist_ave=3):
    np.random.seed(1)
    
    new_seqs = OrderedDict()
    new_key_to_str = OrderedDict()
    for sample_name, seq in seqs.items():
        new_seqs[sample_name] = seq
        label = key_to_str[sample_name]
        
        # uniform random insertion
        if rand_uniform:
            if length > seq.shape[0]:
                augmented_seq = insert_random(seq, length)
            else:
                augmented_seq = delete_random(seq, length)
            augmented_name = sample_name + "_augrand_uniform"
            new_seqs[augmented_name] = augmented_seq
            new_key_to_str[augmented_name] = label
            
        # normal distribution random insertion
        for ave_p in [(i + 1) / (num_normaldist_ave + 1)
                      for i in range(num_normaldist_ave)]:
            ave = (seq.shape[0] - 2) * ave_p
            std = seq.shape[0] * 0.25
            if length > seq.shape[0]:
                augmented_seq = insert_normal_distribution(seq,
                                                           length,
                                                           ave, std)
            else:
                augmented_seq = delete_normal_distribution(seq,
                                                           length,
                                                           ave, std)
            augmented_name = sample_name + "_augrand_normaldist"\
                             + str(ave_p)
            new_seqs[augmented_name] = augmented_seq
            new_key_to_str[augmented_name] = label
    return new_seqs, new_key_to_str

#############################
#         Insert            #
#############################
    
def insert_random(seq, length):
    num = length - seq.shape[0]
    to_insert = np.random.randint(seq.shape[0] - 1, size=num)
    seq = insert_steps_with_balance(seq, to_insert)
    return seq

def insert_normal_distribution(seq, length, ave, std):
    num = length - seq.shape[0]
    to_insert = np.round(truncnorm.rvs((0 - ave) / std,
                                       (seq.shape[0] - 2 - ave) / std,
                                       loc=ave,
                                       scale=std,
                                       size=num))
    seq = insert_steps_with_balance(seq, to_insert)
    return seq

def insert_steps_with_balance(seq, to_insert):
    """
    seq: sequence to augment
    to_insert: array of steps to insert. random, unsorted array 
               is acceptable.
    """
    to_insert = np.array(to_insert).astype(int)
    to_insert = sorted(to_insert)

    count = [0] * (seq.shape[0] - 1)
    for t in to_insert:
        count[t] += 1
        
    for step in range(seq.shape[0] - 1)[::-1]:
        if count[step] != 0:
            seq = insert_steps_between_two_with_balance(seq,
                                                        step,
                                                        count[step])
    return seq

def insert_steps_between_two_with_balance(seq, time_insert, num):
    assert time_insert < seq.shape[0] - 1
    diff = seq[time_insert + 1] - seq[time_insert]
    for i in range(1, num + 1):
        new_step = seq[time_insert] +\
                   diff * i / (num + 1)
        seq = np.insert(seq, time_insert + i, new_step, axis=0)
    return seq

#############################
#         Delete            #
#############################

def delete_random(seq, length):
    num = seq.shape[0] - length
    to_delete = np.random.choice(np.arange(1, seq.shape[0] - 1),
                                 num,
                                 replace=False)
    
    seq = delete_steps(seq, to_delete)
    return seq

def delete_normal_distribution(seq, length, ave, std):
    num = seq.shape[0] - length

    probability_density = []
    for i in range(1, seq.shape[0] - 1):
        probability_density.append(np.round(\
                truncnorm.pdf(i,
                              (1 - ave) / std,
                              (seq.shape[0] - 2 - ave) / std,
                              loc=ave,
                              scale=std)))
    # the sum of probability_density may not just 1 because of error,
    # hence arrange the biggest value
    probability_density[ave] -= (1 - np.sum(probability_density))
    to_delete = np.random.choice(np.arange(1, seq.shape[0] - 1),
                                 num,
                                 p=probability_density,
                                 replace=False)
    
    seq = delete_steps(seq, to_delete)
    return seq

def delete_steps(seq, to_delete):
    to_delete = np.array(sorted(to_delete))
    for step in to_delete[::-1]:
        seq = delete_step(seq, step)
    return seq

def delete_step(seq, time_delete):
    new_seq = np.delete(seq, time_delete, axis=0)
    return new_seq






#############################
#           Main            #
#############################

def main():
    dataset_type = sys.argv[1]
    direc = sys.argv[2]
    path = sys.argv[3]
    length = int(sys.argv[4])
    num_normaldist_ave = 3
    
    seqs, _, _ = read_sequences(dataset_type,
                                direc=direc,
                                feature_normalization=True)
    print(len(seqs))
    seqs = augment_data(seqs, length,
                        num_normaldist_ave=num_normaldist_ave)

    print(len(seqs))
    i = 0
    for sample_name, seq in seqs.items():
        if i % 5 == 0:
            original_name = sample_name
        elif i % 5 == 1:
            dic = dict(augmented_seq=seq,
                       sample_name=sample_name,
                       original_name=original_name,
                       distribution="uniform_random")
            f_path = os.path.join(path, sample_name)
            file_utils.save_pickle(f_path, dic)
        else:
            dic = dict(augmented_seq=seq,
                       sample_name=sample_name,
                       original_name=original_name,
                       distribution="uniform_random")
            f_path = os.path.join(path, sample_name)
            file_utils.save_pickle(f_path, dic)
        i += 1

if __name__ == "__main__":
    main()

    
