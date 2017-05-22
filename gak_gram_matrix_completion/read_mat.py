import numpy as np
import scipy.io as sio
import sys

mat = sio.loadmat(sys.argv[1])
num_to_show = int(sys.argv[2])

for v in mat['gest'].transpose():
    print("[", end=" ")
    for val in v[:num_to_show]:
        print("%10f" % val, end=" ")
    print("]")
