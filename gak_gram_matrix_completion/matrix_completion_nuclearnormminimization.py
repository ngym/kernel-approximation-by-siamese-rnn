import sys, random

import numpy as np
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

def neclearnormminimization_matrix_completion(incomplete_similarities_):
    """
    matrix completion using convex optimization to find low-rank solution
    that still matches observed values. Slow!
    """
    return NuclearNormMinimization().complete(incomplete_similarities_)

def main():
    filename = sys.argv[1]
    incomplete_percentage = int(sys.argv[2])
    errorfile = sys.argv[3]
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']

    seed = 1
        
    incomplete_similarities, dropped_elements = make_matrix_incomplete(seed, similarities, incomplete_percentage)

    html_out_nuclear_norm_minimization = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_NuclearNormMinimization.html")
    mat_out_nuclear_norm_minimization = filename.replace(".mat", "_loss" + str(incomplete_percentage) + "_NuclearNormMinimization.mat")
    
    # "NUCLEAR_NORM_MINIMIZATION"
    completed_similarities = neclearnormminimization_matrix_completion(incomplete_similarities)
    # eigenvalue check
    psd_completed_similarities = nearest_positive_semidefinite(completed_similarities)

    # OUTPUT
    io.savemat(mat_out_nuclear_norm_minimization,
               dict(gram=psd_completed_similarities,
                    dropped_gram=incomplete_similarities,
                    indices=files))
    plot(html_out_nuclear_norm_minimization,
         psd_completed_similarities, files)
    print("NuclearNormMinimization is output")

    mse = mean_squared_error(similarities, psd_completed_similarities)
    msede = mean_squared_error_of_dropped_elements(similarities, psd_completed_similarities, dropped_elements)
    fd = open(errorfile, "w")
    fd.write("Mean squared error: " + str(mse))
    fd.write("\n")
    fd.write("Mean squared error of dropped elements: " + str(msede))
    fd.write("\n")
    fd.close()

if __name__ == "__main__":
    main()
