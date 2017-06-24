import numpy as np

def nearest_positive_semidefinite(matrix):
    w, v = np.linalg.eig(matrix)
    psd_w = np.array([max(0, e) for e in w])
    psd_matrix = np.dot(v, np.dot(np.diag(psd_w), np.linalg.inv(v)))
    return np.real(psd_matrix)

