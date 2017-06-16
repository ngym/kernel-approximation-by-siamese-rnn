import sys, csv
import numpy as np
import scipy as sp
from scipy import io


def main():
    # maybe original matrix and completed matrix are input
    f1 = sys.argv[1]
    f2 = sys.argv[2]
    
    mat1 = io.loadmat(f1)
    mat2 = io.loadmat(f2)
    
    sim1 = mat1['gram']
    sim2 = mat2['gram']

    assert sim1.shape == sim2.shape

    errors = []
    for i in range(sim1.shape[0]):
        for j in range(sim1.shape[1]):
            diff = sim1[i][j] - sim2[i][j]
            if diff != 0:
                errors.append(diff)
    mean_abs_error = np.mean([np.abs(e) for e in errors])
    mean_squared_error = np.mean([e**2 for e in errors])
    print("mean_abs_error: %.10f" % mean_abs_error)
    print("mean_squared_error: %.10f" % mean_squared_error)
    
if __name__ == "__main__":
    main()
