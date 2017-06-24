import numpy as np

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
        mse = np.square(np.linalg.norm(m1[elements[:, 0], elements[:, 1]] - m2[elements[:, 0], elements[:, 1]])) / len(elements)
    else:
        mse = np.square(np.linalg.norm(m1 - m2)) / np.prod(m1.shape)
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
    
