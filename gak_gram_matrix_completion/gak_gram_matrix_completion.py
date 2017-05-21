import numpy as np
import TGA_python3_wrapper.global_align as ga
import scipy.io as sio

import subprocess, functools, sys, threading, glob
import concurrent.futures

import plotly.offline as po
import plotly.graph_objs as pgo

class Logger:
    def __init__(self, log_file):
        self.__lock = threading.Lock()
        self.__fd = open(log_file, 'w')
    def write(self, msg):
        self.__lock.acquire()
        try:
            self.__fd.write(msg)
            self.__fd.flush()
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

def read_mats_and_build_seqs(files):
    for f in files:
        mat = sio.loadmat(f)
        seqs[f] = map_twice(np.float64, mat['gest'].transpose())

def map_twice(func, ll):
    retval = []
    for l in ll:
        retval.append(list(map(func, l[:3])))
    return np.array(retval)

#DATA_DIR = "../../dataset/6DMG_mat_061411/matL/"
DATA_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset/6DMG_mat_112712/matR_char/"

def gak_from_files(f1, f2):
    #print(threading.get_ident())
    """
    mat1 = sio.loadmat(f1)
    mat2 = sio.loadmat(f2)

    def func(x):
        return np.float64(x)
    seq1 = map_twice(func, mat1['gest'].transpose())
    seq2 = map_twice(func, mat2['gest'].transpose())
    """
    seq1 = seqs[f1]
    seq2 = seqs[f2]
    """
    def minus(seq):
        delta = seq[0]
        new_seq = []
        for vector in seq:
            new_v = []
            for i in range(vector.__len__()):
                new_v.append(vector[i]  - delta[i] + 10**6)
            new_seq.append(new_v)
        return new_seq
    seq1 = np.array(minus(seq1))
    seq2 = np.array(minus(seq2))
    """
    #print (seq1)
    #print (seq2)

    T1 = seq1.__len__()
    T2 = seq2.__len__()
    #print(T1)
    #print(T2)

    #sigma = 0.5*(T1+T2)/2*np.sqrt((T1+T2)/2)
    #print("sigma: " + repr(sigma), end="  ")
    sigma = 3000
    Ts = range(10)
    diff_t = np.abs(T1-T2)

    triangular = 0

    val = ga.tga_dissimilarity(seq1, seq2, sigma, triangular)
    kval = np.exp(-val)
    if 0 < triangular <= diff_t:
        # for 0 < triangular <= diff_t, exp(-tga_d) == 0
        assert kval == 0
    #print(f0, end="  ")
    #print(f1, end="  ")
    #print("T=%d \t exp(-tga_d)=%0.5f" % (triangular, kval))
    gak_logger.write(f1 + ", " + f2 + ", " + str(kval) + "\n")
    gram.register(f1, f2, kval)
    return kval

if __name__ == "__main__":
    file_out = sys.argv[1]
    num_thread = int(sys.argv[2])

    gak_logger = Logger("gak_memo.ac")

    files_raw = subprocess.check_output(["ls " + DATA_DIR + "num_4*" + " " + DATA_DIR + "upper_F*"],
                                        universal_newlines=True, shell=True)
    files = files_raw.split('\n')[:-1]
    #print(files)
    """
    files = glob.glob(DATA_DIR + "*.mat")
    print(files)
    """

    read_mats_and_build_seqs(files)
    
    gram = GRAMmatrix(files)

    similarities = []
    file_num = files.__len__()

    futures = []

    def worker(f0index):
        f0 = files[f0index]
        for f1index in range(f0index, file_num):
            f1 = files[f1index]
            #print(f1, end="     ")
            gak_from_files(f0, f1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        for f0index in range(file_num):
            #print(f0, end="  ")
            futures.append(executor.submit(worker, f0index))
                
    concurrent.futures.wait(futures)

    similarities = []
    for i in gram.gram.values():
        similarities.append(list(i.values()))

    trace = pgo.Heatmap(z=similarities,
                        x=files,
                        y=files
    )
    data=[trace]
    po.plot(data, filename=file_out)
