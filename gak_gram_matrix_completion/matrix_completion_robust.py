import sys, random

import scipy as sp
from scipy import io
from scipy.io import wavfile
from scipy import signal

from sklearn.metrics import mean_squared_error

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

from nearest_positive_semidefinite import nearest_positive_semidefinite
from mean_squared_error_of_dropped_elements import mean_squared_error_of_dropped_elements
from plot_gram_matrix import plot
from make_matrix_incomplete import make_matrix_incomplete

import numpy as np
import cvxpy

def robust_matrix_completion(Z, l=None, max_iters=10000, eps=1e-4):
    """Solve Robust Matrix Completion problem via CVXPY with large scale SCS solver.

    :param Z: Input matrix (features x samples)
    :param l: Regularization parameter
    :param max_iters: Iteration count limit
    :param eps: Convergence tolerance
    :type Z: np.ndarray
    :type l: float
    :type max_iters: int
    :type eps: float

    References:
        [1] `Shang, Fanhua, et al. "Robust principal component analysis with missing data."
        <http://www1.se.cuhk.edu.hk/~hcheng/paper/cikm2014fan.pdf>`_.
    """
    Z = np.array(Z)
    assert(isinstance(Z, np.ndarray) and Z.ndim is 2)
    if l is None:
        l = 1 / np.sqrt(np.max(Z.shape))
    assert(l >= 0)
    M = 1 - np.isnan(Z)
    m = np.where(M)
    X = cvxpy.Variable(Z.shape[0], Z.shape[1])
    E = cvxpy.Variable(Z.shape[0], Z.shape[1])
    obj = cvxpy.Minimize( cvxpy.norm(X, 'nuc') + l * cvxpy.norm(E, 1) )
    constraints = [X[m] + E[m] == Z[m]]
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver='SCS', max_iters=max_iters, verbose=True, use_indirect=True, eps=eps)
    X_filled = np.asarray(X.value)
    outliers = np.asarray(E.value)
    return X_filled, outliers

def main():
    filename = sys.argv[1]
    incomplete_percentage = int(sys.argv[2])
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']
    
    seed = 1
        
    incomplete_similarities, dropped_elements = make_matrix_incomplete(seed, similarities, incomplete_percentage)

    html_out_robust_matrix_completion = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RobustMatrixCompletion.html")
    mat_out_robust_matrix_completion = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_RobustMatrixCompletion.mat")
    
    # "SOFT_IMPUTE"
    completed_similarities, _ = robust_matrix_completion(incomplete_similarities)
    # eigenvalue check
    psd_completed_similarities = nearest_positive_semidefinite(completed_similarities)

    # OUTPUT
    io.savemat(mat_out_robust_matrix_completion,
               dict(gram=psd_completed_similarities,
                    dropped_gram=incomplete_similarities,
                    indices=files))
    plot(html_out_robust_matrix_completion,
         psd_completed_similarities, files)
    print("RobustMatrixCompletion is output")

    mse = mean_squared_error(similarities, psd_completed_similarities)
    print("Mean squared error: " + str(mse))
    msede = mean_squared_error_of_dropped_elements(similarities, psd_completed_similarities, dropped_elements)
    print("Mean squared error of dropped elements: " + str(msede))

if __name__ == "__main__":
    main()
