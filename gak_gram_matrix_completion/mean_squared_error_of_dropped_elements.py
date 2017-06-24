import numpy as np
from sklearn.metrics import mean_squared_error

def mean_squared_error_of_dropped_elements(m1, m2, elements):
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    elements = np.array(elements)
    assert elements.shape[1] == 2
    mse = np.linalg.norm(m1[elements[:, 0], elements[:, 1]] -
                         m2[elements[:, 0], elements[:, 1]]) / len(elements)
    return mse

def relative_error(m1, m2, elements):
    # m1 is ground truth
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    elements = np.array(elements)
    assert elements.shape[1] == 2
    re = np.linalg.norm(m1[elements[:, 0], elements[:, 1]] - m2[elements[:, 0], elements[:, 1]])\
          / np.linalg.norm(m1[elements[:, 0], elements[:, 1]])
    return re
    
    
