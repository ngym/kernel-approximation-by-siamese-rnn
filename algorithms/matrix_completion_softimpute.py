import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
from scipy import io
from fancyimpute import SoftImpute

from utils.nearest_positive_semidefinite import nearest_positive_semidefinite
from utils.errors import mean_squared_error
from utils.plot_gram_to_html import plot_gram_to_html
from utils.make_matrix_incomplete import gram_drop_random


def softimpute_matrix_completion(gram_drop):
    """Fill in Gram matrix with dropped elements with Soft Impute Matrix Completion.
    Optimizes the Matrix Completion objective using Singular Value Thresholding

    :param gram_drop: Gram matrix with dropped elements
    :type gram_drop: list of lists
    :returns: Filled in Gram matrix, optimization start and end times
    :rtype: list of lists, float, float, float, float
    """
    t_start = time.time()
    gram_completed = SoftImpute().complete(gram_drop)
    t_end = time.time()
    return gram_completed, t_start, t_end
    
def main():
    filename = sys.argv[1]
    drop_percent = int(sys.argv[2])
    completionanalysisfile = sys.argv[3]
    mat = io.loadmat(filename)
    gram = mat['gram']
    files = mat['indices']

    seed = 1
        
    fd = open(completionanalysisfile, "w")
    
    gram_drop, dropped_elements = gram_drop_random(seed, gram, drop_percent)

    fd.write("number of dropped elements: " + str(len(dropped_elements)))
    fd.write("\n")

    html_out_soft_impute = filename.replace(".mat", "_drop" + str(drop_percent) + "_SoftImpute.html")
    mat_out_soft_impute = filename.replace(".mat", "_drop" + str(drop_percent) + "_SoftImpute.mat")

    # Soft Impute 
    gram_completed, t_start, t_end = softimpute_matrix_completion(gram_drop)
    # eigenvalue check
    npsd_start = time.time()
    gram_completed_npsd = nearest_positive_semidefinite(gram_completed)
    npsd_end = time.time()

    # OUTPUT
    io.savemat(mat_out_soft_impute,
               dict(gram=gram_completed_npsd,
                    dropped_gram=gram_drop,
                    indices=files))
    plot_gram_to_html(html_out_soft_impute,
                      gram_completed_npsd, files)
    print("SoftImpute is output")

    mse = mean_squared_error(gram, gram_completed_npsd)
    msede = mean_squared_error(gram, gram_completed_npsd, dropped_elements)
    fd.write("start: " + str(t_start))
    fd.write("\n")
    fd.write("end: " + str(t_end))
    fd.write("\n")
    fd.write("duration: " + str(t_end - t_start))
    fd.write("\n")
    fd.write("Mean squared error: " + str(mse))
    fd.write("\n")
    fd.write("Mean squared error of dropped elements: " + str(msede))
    fd.write("\n")
    fd.close()

if __name__ == "__main__":
    main()

