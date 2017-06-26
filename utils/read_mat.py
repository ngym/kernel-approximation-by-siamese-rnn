"""Print .mat file contents for 6DMG data set.
"""

import numpy as np
from scipy import io
import sys

mat = io.loadmat(sys.argv[1])
num_to_show = int(sys.argv[2])

for v in mat['gest'].transpose():
    print("[", end=" ")
    for val in v[:num_to_show]:
        print("%10f" % val, end=" ")
    print("]")
print(mat['gest'].T.__len__())
