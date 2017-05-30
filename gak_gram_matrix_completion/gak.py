import TGA_python3_wrapper.global_align as ga
import numpy as np

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

