import numpy as np


def mean_squared_error(m1, m2, indices=None):
    """Compute (masked) Mean Squared Error.
    Take subset of elements as vector, compute norm of difference, divided by subset size
    
    :param m1: True matrix
    :param m2: Estimated matrix
    :param indices: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type indices: None or list of 2-tuples
    :return: Mean Squared Error
    :rtype: float
    """
    
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    if indices is not None:
        indices = np.array(indices)
        assert indices.shape[1] == 2
        mse = np.sum(np.square(m1[indices[:, 0], indices[:, 1]] - m2[indices[:, 0], indices[:, 1]])) / len(indices)
    else:
        mse = np.sum(np.square(m1 - m2)) / np.prod(m1.shape)
    return mse


def relative_error(m1, m2, indices=None):
    """Compute (masked) Relative 2-norm Error.
    Take subset of elements as vector, compute norm of difference, divided by norm
    
    :param m1: True matrix
    :param m2: Estimated matrix
    :param indices: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type indices: None or list of 2-tuples
    :return: Relative 2-norm Error
    :rtype: float
    """

    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    if indices is not None:
        indices = np.array(indices)
        assert indices.shape[1] == 2
        re = np.sqrt(np.sum(np.square(m1[indices[:, 0], indices[:, 1]] - m2[indices[:, 0], indices[:, 1]]))) \
              / np.sqrt(np.sum(np.square(m1[indices[:, 0], indices[:, 1]])) + 1e-8)
    else:
        re = np.sqrt(np.sum(np.square(m1 - m2))) / np.sqrt(np.sum(np.square(m1)) + 1e-8)
    return re


def mean_absolute_error(m1, m2, indices=None):
    """Compute (masked) Mean Absolute Error.
    Take subset of elements as vector, compute norm of difference, divided by subset size

    :param m1: True matrix
    :param m2: Estimated matrix
    :param indices: Subset indices
    :type m1: np.ndarray
    :type m2: np.ndarray
    :type indices: None or list of 2-tuples
    :return: Mean Absolute Error
    :rtype: float
    """
    
    m1 = np.array(m1)
    m2 = np.array(m2)
    assert m1.shape == m2.shape
    if indices is not None:
        indices = np.array(indices)
        assert indices.shape[1] == 2
        mae = np.sum(np.abs(m1[indices[:, 0], indices[:, 1]] - m2[indices[:, 0], indices[:, 1]])) / len(indices)
    else:
        mae = np.sum(np.abs(m1 - m2)) / np.prod(m1.shape)
    return mae