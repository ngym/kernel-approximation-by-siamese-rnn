import numpy as np

def mean_squared_error_of_dropped_elements(m1, m2, elements):
    """Compute masked Mean Squared Error.
    Take subset of elements as vector, compute norm of difference, divided by subset size
    
    :param m1: True matrix
    :param m2: Estimated matrix
    :param elements: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type elements: list of 2-tuples
    :returns: Mean Squared Error
    :rtype: float
    """
    
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    elements = np.array(elements)
    assert elements.shape[1] == 2
    mse = np.square(np.linalg.norm(m1[elements[:, 0], elements[:, 1]] - m2[elements[:, 0], elements[:, 1]])) / len(elements)
    return mse

def relative_error(m1, m2, elements):
    """Compute masked Relative 2-norm Error.
    Take subset of elements as vector, compute norm of difference, divided by norm
    
    :param m1: True matrix
    :param m2: Estimated matrix
    :param elements: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type elements: list of 2-tuples
    :returns: Relative Squared Error
    :rtype: float
    """

    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    elements = np.array(elements)
    assert elements.shape[1] == 2
    re = np.linalg.norm(m1[elements[:, 0], elements[:, 1]] - m2[elements[:, 0], elements[:, 1]]) \
          / np.linalg.norm(m1[elements[:, 0], elements[:, 1]])
    return re
    
    
