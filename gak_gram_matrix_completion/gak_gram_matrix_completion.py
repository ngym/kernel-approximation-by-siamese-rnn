import numpy as np
import TGA_python3_wrapper.global_align as ga
import scipy.io as sio

import subprocess, functools, sys, threading, glob, json, random
import concurrent.futures

import plotly.offline as po
import plotly.graph_objs as pgo

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute

class Logger:
    def __init__(self, log_file):
        self.__lock = threading.Lock()
        self.__fd = open(log_file, 'w')
    def write(self, msg):
        self.__lock.acquire()
        try:
            self.__fd.write(msg)
            #self.__fd.flush()
        finally:
            self.__lock.release()
    def __del__(self):
        self.__fd.write("\n")
        self.__fd.close()

gak_logger = None

class GRAMmatrix:
    def __init__(self, seq_ids):
        self.__lock = threading.Lock()
        self.gram = {}
        for seq_id1 in seq_ids:
            self.gram[seq_id1] = {}
            for seq_id2 in seq_ids:
                self.gram[seq_id1][seq_id2] = -1
    def register(self, seq_id1, seq_id2, value):
        self.__lock.acquire()
        try:
            self.gram[seq_id1][seq_id2] = value
            self.gram[seq_id2][seq_id1] = value
        finally:
            self.__lock.release()

gram = None

seqs = {}

def read_mats_and_build_seqs(files, attribute_type):
    for f in files:
        mat = sio.loadmat(f)
        seqs[f] = second_map(np.float64, pick_attribute(mat['gest'].transpose(), attribute_type))

def pick_attribute(ll, attribute_type):
    retval = []
    if attribute_type == "position":
        for l in ll:
            retval.append(l[1:4])
    elif attribute_type == "velocity":
        for i in range(ll.__len__() - 1):
            retval.append([(ll[i+1][1] - ll[i][1])/((ll[i+1][0] - ll[i][0])/1000),
                           (ll[i+1][2] - ll[i][2])/((ll[i+1][0] - ll[i][0])/1000),
                           (ll[i+1][3] - ll[i][3])/((ll[i+1][0] - ll[i][0])/1000)])
    elif attribute_type == "acceleration":
        for l in ll:
            retval.append(l[8:11])
    elif attribute_type == "angularvelocity":
        for l in ll:
            retval.append(l[11:14])
    elif attribute_type == "orientation":
        for l in ll:
            retval.append(l[4:8])
    else:
        print("attribute type error.")
        assert False
    return retval
        
def second_map(func, ll):
    retval = []
    for l in ll:
        retval.append(list(map(func, l)))
    return np.array(retval)

def gak(seq1, seq2, sigma):
    #print(threading.get_ident())
    if seq1 is seq2:
        return 1
    
    T1 = seq1.__len__()
    T2 = seq2.__len__()

    #sigma = 0.5*(T1+T2)/2*np.sqrt((T1+T2)/2)
    #sigma = 2 ** 0
    #print("sigma: " + repr(sigma), end="  ")
    
    diff_t = np.abs(T1-T2)

    triangular = 0

    val = ga.tga_dissimilarity(seq1, seq2, sigma, triangular)
    kval = np.exp(-val)
    if 0 < triangular <= diff_t:
        assert kval == 0
    return kval

def plot(file_name, similarities, files_to_show):
    # To fix the direction of the matrix as the diagonal line is from top-left to bottom-right.
    similarities_ = similarities[::-1]
    files_to_show = []
    for f in files:
        files_to_show.append(f.split('/')[-1].split('.')[0])
    files_to_show_ = files_to_show[::-1]
    
    trace = pgo.Heatmap(z=similarities_,
                        x=files_to_show,
                        y=files_to_show_,
                        zmin=0, zmax=1
    )
    data=[trace]
    po.plot(data, filename=file_name, auto_open=False)

if __name__ == "__main__":
    config_json_file = sys.argv[1]
    config_dict = json.load(open(config_json_file, 'r'))
    
    num_thread = config_dict['num_thread']

    dataset_type = config_dict['dataset_type']
    data_attribute_type = config_dict['data_attribute_type']
    
    data_files = config_dict['data_mat_files']
    gak_sigma = np.float64(config_dict['gak_sigma'])
    random_seed = int(config_dict['random_seed'])
    incomplete_persentage = int(config_dict['incomplete_persentage'])

    output_dir = config_dict['output_dir']
    
    output_filename_format = config_dict['output_filename_format'].replace("${dataset_type}", dataset_type)\
                                                                  .replace("${data_attribute_type}", data_attribute_type)\
                                                                  .replace("${gak_sigma}", ("%.3f" % gak_sigma))\
                                                                  .replace("${incomplete_persentage}", str(incomplete_persentage))

    gak_logfile = output_dir + output_filename_format.replace("_${completion_alg}", "") + ".log"
    
    html_out_no_completion = output_dir + output_filename_format.replace("${completion_alg}", "NoCompletion") + ".html" 
    html_out_nuclear_norm_minimization = output_dir + output_filename_format.replace("${completion_alg}", "NuclearNormMinimization") + ".html"
    html_out_soft_impute = output_dir + output_filename_format.replace("${completion_alg}", "SoftImpute") + ".html"
    mat_out_no_completion = output_dir + output_filename_format.replace("${completion_alg}", "NoCompletion") + ".mat" 
    mat_out_nuclear_norm_minimization = output_dir + output_filename_format.replace("${completion_alg}", "NuclearNormMinimization") + ".mat"
    mat_out_soft_impute = output_dir + output_filename_format.replace("${completion_alg}", "SoftImpute") + ".mat"
    
    gak_logger = Logger(gak_logfile)

    files = []
    for df in data_files:
        files_ = glob.glob(df)
        print(files_[:3])
        print("...")
        print(files_[-3:])
        files += files_
    files = sorted(files)

    read_mats_and_build_seqs(files, data_attribute_type)

    gram = GRAMmatrix(files)

    similarities = []
    file_num = files.__len__()

    futures = []

    def worker_for_f1(f1index, f2indices, gak_sigma):
        f1 = files[f1index]
        seq1 = seqs[f1]
        ret_dict = {}
        for f2index in f2indices:
            f2 = files[f2index]
            seq2 = seqs[f2]
            ret_dict[f2] = gak(seq1, seq2, gak_sigma)
        return ret_dict
    #seqs[files[f2index]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_thread) as executor:
        print("Start submitting jobs.")
        future_to_files = {executor.submit(worker_for_f1, f1index,
                                           range(f1index, file_num), gak_sigma):
                           files[f1index]
                           for f1index in range(file_num)}
        num_futures = future_to_files.__len__()
        print(str(num_futures) + " jobs are submitted.")
        num_finished_jobs = 0
        for future in concurrent.futures.as_completed(future_to_files):
            f1 = future_to_files[future]
            ret_dict = future.result()
            ret_dict_keys = list(ret_dict.keys())
            for f2 in ret_dict_keys:
                value = ret_dict[f2]
                gram.register(f1, f2, value)
                #gak_logger.write(f1 + ", " + f2 + ", " + str(value) + "\n")
            num_finished_jobs += 1
            print(str(num_finished_jobs) + "/" + str(num_futures), end=" ")
            sys.stdout.flush()

    print(" ")

    similarities = []
    for i in gram.gram.values():
        similarities.append(list(i.values()))

    # "NO_COMPLETION"
    plot(html_out_no_completion,
         similarities, files)
    sio.savemat(mat_out_no_completion, dict(gram=similarities, indices=files))


    ###################################
    ###### completed GRAM matrix ######
    ###################################
    
    random.seed(random_seed)
        
    incomplete_similarities = []
    for s_row in similarities:
        is_row = []
        for s in s_row:
            if s == 1:
                if similarities.index(s_row) == s_row.index(s):
                    is_row.append(1)
                    continue
            if random.randint(0, 99) < incomplete_persentage:
                is_row.append(np.nan)
            else:
                is_row.append(s)
        incomplete_similarities.append(is_row)

    def check_and_modify_eigenvalues_to_positive_definite(A):
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
        w, v = np.linalg.eig(matrix)
        new_w = np.array([max(0, e) for e in w])
        new_matrix = np.dot(v, np.dot(np.diag(w), np.linalg.inv(v)))
        return np.real(new_matrix)
    """
    
    # "NUCLEAR_NORM_MINIMIZATION":
    """
    matrix completion using convex optimization to find low-rank solution
    that still matches observed values. Slow!
    """
    completed_similarities = NuclearNormMinimization().complete(incomplete_similarities)
    # eigenvalue check
    positive_definite_completed_similarities = check_and_modify_eigenvalues_to_positive_definite(completed_similarities)
    plot(html_out_nuclear_norm_minimization,
         positive_definite_completed_similarities, files)
    sio.savemat(mat_out_nuclear_norm_minimization, dict(gram=completed_similarities, indices=files))

    # "SOFT_IMPUTE"
    """
    Instead of solving the nuclear norm objective directly, instead
    induce sparsity using singular value thresholding
    """
    completed_similarities = SoftImpute().complete(incomplete_similarities)
    # eigenvalue check
    positive_definite_completed_similarities = check_and_modify_eigenvalues_to_positive_definite(completed_similarities)
    plot(html_out_soft_impute,
         positive_definite_completed_similarities, files)
    sio.savemat(mat_out_soft_impute, dict(gram=completed_similarities, indices=files))
 

    """
    loaded_mat = sio.loadmat(mat_out_nuclear_norm_minimization)
    print(loaded_mat['gram'])
    mat = second_map(np.float64, loaded_mat['gram'])
    print(mat)
    plot("tmp.html", mat, loaded_mat['indices'])
    """
