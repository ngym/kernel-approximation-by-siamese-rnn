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
