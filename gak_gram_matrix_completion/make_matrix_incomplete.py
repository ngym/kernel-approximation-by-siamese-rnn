import random
import numpy as np

def make_matrix_incomplete(seed, similarities, incomplete_percentage):
    random.seed(seed)
        
    incomplete_similarities = []
    dropped_elements = []
    for i in range(similarities.__len__()):
        s_row = similarities[i]
        is_row = []
        for j in range(s_row.__len__()):
            if i == j:
                is_row.append(1)
                continue
            if random.randint(0, 99) < incomplete_percentage:
                is_row.append(np.nan)
                dropped_elements.append((i, j))
            else:
                is_row.append(s_row[j])
        incomplete_similarities.append(is_row)
    print("Incomplete matrix is provided.")
    return incomplete_similarities, dropped_elements
