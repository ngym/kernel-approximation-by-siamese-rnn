import sys, os
import os.path as path

from scipy import io

def main():
    filename = sys.argv[1]
    mat = io.loadmat(filename)
    similarities = mat['gram']
    files = mat['indices']

    def cond(f):
        if f.find("t01.mat") != -1 or \
           f.find("t02.mat") != -1 or \
           f.find("t03.mat") != -1:
            return True
        else:
            return False
    
    num = len(files)
    picked_similarities = [
        [similarities[j_row][i_column]
         for i_column in range(num)
         if cond(files[i_column])]
        for j_row in range(num)
        if cond(files[j_row])
    ]
    picked_files = [f for f in files if cond(f)]

    pdf_out = filename.replace(".mat", "_t1-t3.pdf")
    mat_out  = filename.replace(".mat", "_t1-t3.mat")

    io.savemat(mat_out, dict(gram=picked_similarities,
                             indices=picked_files))

if __name__ == "__main__":
    main()
