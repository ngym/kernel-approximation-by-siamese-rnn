import numpy as np
import scipy.io as sio
import sys

mat = sio.loadmat(sys.argv[1])
for v in mat['gest'].transpose():
    print(v[:3])
