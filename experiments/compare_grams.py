import experiments
from experiments.complete_matrix import calculate_errors
from utils import file_utils
import sys

def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    loaded_data1 = file_utils.load_hdf5(file1)
    loaded_data2 = file_utils.load_hdf5(file2)
    assert loaded_data1['dropped_indices'] == loaded_data2['dropped_indices']
    dropped_indices = loaded_data1['dropped_indices']
    dropped_elements = [(i, j) for i in dropped_indices for j in dropped_indices]
    
    gram1 = loaded_data1['gram_matrices'][-1]['completed_npsd']
    gram2 = loaded_data2['gram_matrices'][-1]['completed_npsd']
    
    errs = calculate_errors(gram1, gram2, dropped_elements)
    print("mse:%f, mse_dropped:%f, mae:%f, mae_dropped:%f, re:%f, re_dropped:%f" % errs)


if __name__ == "__main__":
    main()


    
