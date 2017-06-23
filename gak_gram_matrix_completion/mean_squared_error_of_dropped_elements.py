import numpy as np

def mean_squared_error_of_dropped_elements(m1, m2, elements):
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    sum_squared_error = 0
    for i, j in elements:
        sum_squared_error += (m1[i][j] - m2[i][j]) ** 2
    return sum_squared_error / elements.__len__()

def mean_ratio_between_absolute_loss_and_absolute_true_value_of_dropped_elements(m1, m2, elements):
    # m1 is ground truth
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    ratio_between_absolute_loss_and_absolute_true_value = 0    
    for i, j in elements:
        ratio_between_absolute_loss_and_absolute_true_value +=\
            np.abs((m1[i][j] - m2[i][j])) / m1[i][j]
    return ratio_between_absolute_loss_and_absolute_true_value / elements.__len__()
    
    
