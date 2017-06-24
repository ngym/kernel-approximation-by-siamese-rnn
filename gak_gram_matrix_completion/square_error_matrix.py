"""Read two .mat files and compute Mean Squared Error between their Gram matrices.
"""

import sys, random

import plotly.offline as po
import plotly.graph_objs as pgo

import numpy as np
import scipy as sp
from scipy import io
from scipy.io import wavfile
from scipy import signal

from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    f1 = sys.argv[1]
    f2 = sys.argv[2]

    mat1 = io.loadmat(f1)
    mat2 = io.loadmat(f2)

    gram1 = mat1['gram']
    gram2 = mat2['gram']

    mse = mean_squared_error(gram1, gram2)

    print("Mean squared error: " + str(mse))
    

    
