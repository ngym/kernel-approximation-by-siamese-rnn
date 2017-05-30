import numpy as np

def mean_squared_error_of_dropped_elements(m1, m2, elements):
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    sum_squared_error = 0
    for i, j in elements:
        sum_squared_error += (m1[i][j] - m2[i][j]) ** 2
    return sum_squared_error / elements.__len__()

