import sys

import numpy as np
import scipy as sp
from scipy import io

def mean_squared_error(m1, m2, elements=None):
    """Compute (masked) Mean Squared Error.
    Take subset of elements as vector, compute norm of difference, divided by subset size
    
    :param m1: True matrix
    :param m2: Estimated matrix
    :param elements: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type elements: None or list of 2-tuples
    :returns: Mean Squared Error
    :rtype: float
    """
    
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    if elements is not None:
        elements = np.array(elements)
        assert elements.shape[1] == 2
        mse = np.sum(np.square(m1[elements[:, 0], elements[:, 1]] - m2[elements[:, 0], elements[:, 1]])) / len(elements)
    else:
        mse = np.sum(np.square(m1 - m2)) / np.prod(m1.shape)
    return mse

def relative_error(m1, m2, elements=None):
    """Compute (masked) Relative 2-norm Error.
    Take subset of elements as vector, compute norm of difference, divided by norm
    
    :param m1: True matrix
    :param m2: Estimated matrix
    :param elements: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type elements: None or list of 2-tuples
    :returns: Relative 2-norm Error
    :rtype: float
    """

    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    if elements is not None:
        elements = np.array(elements)
        assert elements.shape[1] == 2
        re = np.linalg.norm(m1[elements[:, 0], elements[:, 1]] - m2[elements[:, 0], elements[:, 1]]) \
              / np.linalg.norm(m1[elements[:, 0], elements[:, 1]])
    else:
        re = np.linalg.norm(m1 - m2) / np.linalg.norm(m1)
    return re
    
def mean_absolute_error(m1, m2, elements=None):
    """Compute (masked) Mean Absolute Error.
    Take subset of elements as vector, compute norm of difference, divided by subset size
    
    :param m1: True matrix
    :param m2: Estimated matrix
    :param elements: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type elements: None or list of 2-tuples
    :returns: Mean Absolute Error
    :rtype: float
    """
    
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    if elements is not None:
        elements = np.array(elements)
        assert elements.shape[1] == 2
        mse = np.sum(np.abs(m1[elements[:, 0], elements[:, 1]] - m2[elements[:, 0], elements[:, 1]])) / len(elements)
    else:
        mae = np.sum(np.abs(m1 - m2)) / np.prod(m1.shape)
    return mae

def main():
    f = sys.argv[1]

    mat = io.loadmat(f)

    gram1 = mat['orig_gram']
    gram2 = mat['gram']

    assert gram1.shape == gram2.shape

    dropped_elements = []
    for i in range(len(gram1)):
        for j in range(len(gram1)):
            if gram1[i][j] != gram2[i][j]:
                dropped_elements.append((i, j))
    
    mse = mean_squared_error(gram1, gram2)
    print("Mean squared error: %.10f" % mse)
    msede = mean_squared_error(gram1, gram2, dropped_elements)
    print("Mean squared error of dropped elements: %.10f" msede)

    mae = mean_absolute_error(gram1, gram2)
    print("Mean absolute error: %.10f" % mae)
    maede = mean_absolute_error(gram1, gram2, dropped_elements)
    print("Mean absolute error of dropped elements: %.10f" maede)
    
    re = relative_error(gram1, gram2)
    print("Relative error: %.10f" % re)
    rede = relative_error(gram1, gram2, dropped_elements)
    print("Relative error of dropped elements: %.10f" % rede)
    

if __name__ == "__main__":
    main()
