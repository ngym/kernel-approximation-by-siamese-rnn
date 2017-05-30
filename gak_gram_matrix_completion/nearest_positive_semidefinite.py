import numpy as np

def nearest_positive_semidefinite(matrix):
    sym_matrix = (matrix + matrix.T) * 0.5
    w, v = np.linalg.eig(sym_matrix)
    psd_w = np.array([max(0, e) for e in w])
    psd_matrix = np.dot(v, np.dot(np.diag(psd_w), np.linalg.inv(v)))
    return np.real(psd_matrix)

    """
        epsilon=0
        n = A.shape[0]
        eigval, eigvec = np.linalg.eig(A)
        val = np.matrix(np.maximum(eigval,epsilon))
        vec = np.matrix(eigvec)
        T = 1/(np.multiply(vec,vec) * val.T)
        T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
        B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
        out = B*B.T
        return(np.real(out))
    """
