import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from experiments.complete_matrix import calculate_errors
from utils import file_utils

def main():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    loaded_data1 = file_utils.load_hdf5(file1)
    loaded_data2 = file_utils.load_hdf5(file2)
    assert loaded_data1['dropped_indices'] == loaded_data2['dropped_indices']
    dropped_indices = loaded_data1['dropped_indices'][-1]
    dropped_elements = [(i, j) for i in dropped_indices for j in dropped_indices]
    
    gram1 = loaded_data1['gram_matrices'][-1]['completed_npsd']
    gram2 = loaded_data2['gram_matrices'][-1]['completed_npsd']
    
    errs = calculate_errors(gram1, gram2, dropped_elements)
    print("mse:%.10f, mse_dropped:%.10f, mae:%.10f, mae_dropped:%.10f, re:%.10f, re_dropped:%.10f" % errs)
    

if __name__ == "__main__":
    main()


    
