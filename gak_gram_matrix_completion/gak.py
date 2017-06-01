import sys
import TGA_python3_wrapper.global_align as ga
import numpy as np

import scipy as sp
from scipy import io
from scipy.io import wavfile
from scipy import signal

def gak(seq1, seq2, sigma, triangular):
    #print(threading.get_ident())
    if seq1 is seq2:
        return 1
    
    T1 = seq1.__len__()
    T2 = seq2.__len__()

    #sigma = 0.5*(T1+T2)/2*np.sqrt((T1+T2)/2)
    #sigma = 2 ** 0
    #print("sigma: " + repr(sigma), end="  ")
    
    #triangular = (T1+T2) * 0.5

    val = ga.tga_dissimilarity(seq1, seq2, sigma, triangular)
    kval = np.exp(-val)
    return kval

def read_and_resample(f, frequency):
    mat_filename = f.replace(".wav", ("_freq" + str(frequency) + ".mat"))
    print(mat_filename)
    try:
        mat = io.loadmat(mat_filename)
        resampled_data = mat['resampled_data']
        print("read from mat")
    except:
        rate, data = io.wavfile.read(f)
        length = frequency * data.__len__() // rate
        print("read from wav")
        resampled_data = signal.resample(data, length)
    return resampled_data

def main():
    f1 = sys.argv[1]
    f2 = sys.argv[2]

    seq1 = read_and_resample(f1, 100)
    seq2 = read_and_resample(f2, 100)

    val = gak(seq1, seq2, 0.4, 500)
    print(val)


if __name__ == "__main__":
    main()
