import sys
import numpy as np
import scipy as sp
from scipy import io

import matplotlib.pyplot as plt

def plot(file_name, similarities, files):
    similarities_ = similarities[::-1]
    files_to_show = []
    for f in files:
        files_to_show.append(f.split('/')[-1].split('.')[0])
    files_to_show_ = files_to_show[::-1]

    f = plt.imshow(similarities, interpolation='nearest')
    plt.savefig(file_name,  format='pdf', dpi=1200)

def main():
    filename = sys.argv[1]
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']
    seqs = {}

    seed = 1
        
    png_out = filename.replace(".mat", ".pdf")
    
    # OUTPUT
    plot(png_out,
         similarities, files)

if __name__ == "__main__":
    main()
