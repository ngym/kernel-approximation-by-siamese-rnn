import numpy as np

def nearest_positive_semidefinite(sym_matrix):
    """Compute the closest positive semidefinite matrix to a given symmetric matrix.
    Compute eigen decomposition, zero out negative eigenvalues
    
    :param sym_matrix: Symmetric matrix to be approximated
    :type sym_matrix: np.ndarray
    :returns: Positive semidefinite matrix approximation
    :rtype: np.ndarray
    """
    
    assert(np.allclose(sym_matrix, sym_matrix.T, atol=1e-05))
    w, v = np.linalg.eig(sym_matrix)
    psd_w = np.array([max(0, e) for e in w])
    psd_matrix = np.dot(v, np.dot(np.diag(psd_w), np.linalg.inv(v)))
    return np.real(psd_matrix)

