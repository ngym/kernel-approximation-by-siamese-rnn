import sys, json, glob, os

from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
import sklearn.metrics as metrics
import scipy.io as sio
import numpy as np
import functools

from matrix_completion_softimpute import softimpute_matrix_completion

from make_matrix_incomplete import make_matrix_incomplete, drop_randomly_in_samples

def main():
    filename = sys.argv[1]
    incomplete_percentage = int(sys.argv[2])
    completionanalysisfile = sys.argv[3]
    mat = io.loadmat(filename)
    similarities = mat['gram']

    indices_drop_from = mat['dropped_indices_number']

    seed = 1

    fd = open(completionanalysisfile, "w")

    incomplete_similarities, \
        dropped_elements = drop_randomly_in_samples(seed, gram, indices_drop_from,
                                                    loss_persentage)

    fd.write("number of dropped elements: " + str(len(dropped_elements)))
    fd.write("\n")

    html_out_soft_impute = filename.replace(".mat",
                                            "_loss" + str(incomplete_percentage) + \
                                            "_SoftImpute.html")
    mat_out_soft_impute = filename.replace(".mat",
                                           "_loss" + str(incomplete_percentage) + \
                                           "_SoftImpute.mat")

    t_start = time.time()
    # "SOFT_IMPUTE"
    completed_similarities = softimpute_matrix_completion(incomplete_similarities)
    # eigenvalue check
    psd_completed_similarities = nearest_positive_semidefinite(completed_similarities)
    t_finish = time.time()

    # OUTPUT
    io.savemat(mat_out_soft_impute,
               dict(gram=psd_completed_similarities,
                    dropped_gram=incomplete_similarities,
                    indices=files))
    plot(html_out_soft_impute,
         psd_completed_similarities, files)
    print("SoftImpute is output")

    mse = mean_squared_error(similarities, psd_completed_similarities)
    msede = mean_squared_error_of_dropped_elements(similarities,
                                                   psd_completed_similarities,
                                                   dropped_elements)
    fd.write("start: " + str(t_start))
    fd.write("\n")
    fd.write("finish: " + str(t_finish))
    fd.write("\n")
    fd.write("duration: " + str(t_finish - t_start))
    fd.write("\n")
    fd.write("Mean squared error: " + str(mse))
    fd.write("\n")
    fd.write("Mean squared error of dropped elements: " + str(msede))
    fd.write("\n")
    fd.close()

if __name__ == '__main__':
    main()

