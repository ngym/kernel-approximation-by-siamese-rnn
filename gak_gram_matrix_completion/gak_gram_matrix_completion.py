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

def read_mats_and_build_seqs(files):
    for f in files:
        mat = sio.loadmat(f)
        seqs[f] = map_twice(np.float64, mat['gest'].transpose())

def map_twice(func, ll):
    retval = []
    for l in ll:
        retval.append(list(map(func, l[:3])))
    return np.array(retval)

DATA_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/dataset/6DMG_mat_112712/matR_char/"

def gak_from_files(f1, f2):
    #print(threading.get_ident())
    
    seq1 = seqs[f1]
    seq2 = seqs[f2]

    T1 = seq1.__len__()
    T2 = seq2.__len__()

    #sigma = 0.5*(T1+T2)/2*np.sqrt((T1+T2)/2)
    #print("sigma: " + repr(sigma), end="  ")
    sigma = 3000
    Ts = range(10)
    diff_t = np.abs(T1-T2)

    triangular = 0

    val = ga.tga_dissimilarity(seq1, seq2, sigma, triangular)
    kval = np.exp(-val)
    if 0 < triangular <= diff_t:
        assert kval == 0
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_thread) as executor:
        future_to_files = {executor.submit(gak_from_files, files[f1index], files[f2index]):
                           (files[f1index], files[f2index])
                           for f1index in range(file_num)
                           for f2index in range(f1index, file_num)}
        for future in concurrent.futures.as_completed(future_to_files):
            f1, f2 = future_to_files[future]
            value = future.result()
            gram.register(f1, f2, value)
            gak_logger.write(f1 + ", " + f2 + ", " + str(value) + "\n")
                
    similarities = []
    for i in gram.gram.values():
        similarities.append(list(i.values()))

    trace = pgo.Heatmap(z=similarities,
                        x=files,
                        y=files
    )
    data=[trace]
    po.plot(data, filename=file_out)
