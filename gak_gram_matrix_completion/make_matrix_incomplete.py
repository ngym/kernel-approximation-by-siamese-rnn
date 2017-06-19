import random
import numpy as np
import copy

def make_matrix_incomplete(seed, similarities, incomplete_percentage):
    random.seed(seed)
        
    incomplete_similarities = []
    dropped_elements = []

    half_indices = [(i, j) for i in range(len(similarities))
                   for j in range(i + 1, len(similarities[0]))]
    permutated_half_indices = np.random.permutation(half_indices)
    num_to_drop = int(len(permutated_half_indices) * incomplete_percentage * 0.01)
    dropped_elements = permutated_half_indices[:num_to_drop]

    incomplete_similarities = copy.deepcopy(similarities)
    for i, j in dropped_elements:
        incomplete_similarities[i][j] = np.nan
        incomplete_similarities[j][i] = np.nan
        
    print("Incomplete matrix is provided.")
    return incomplete_similarities, dropped_elements

def drop_samples(similarities, indices):
    dropped_elements = []
    for index in indices:
        similarities, dropped_elements_ = drop_one_sample(similarities, index)
        dropped_elements += dropped_elements_
    return similarities, dropped_elements

def drop_one_sample(similarities, index):
    incomplete_similarities = copy.deepcopy(similarities)
    dropped_elements = []
    for i in range(len(similarities)):
        incomplete_similarities[i][index] = np.nan
        dropped_elements.append((i, index))
    for j in range(len(similarities[0])):
        incomplete_similarities[index][j] = np.nan
        dropped_elements.append((index, j))

    print("Incomplete matrix is provided.")
    return incomplete_similarities, dropped_elements
    
