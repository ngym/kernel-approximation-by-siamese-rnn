import os, sys, shutil, json
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import save_json

model_dirs =[
    ("6DMG", "results/linear_on_branch/tanh_in_LSTM/6DMG/20/t1-t3/10/10,10/0.3/mse/10.0/dot_product/"),
    ("6DMG", "results/linear_on_branch/tanh_in_LSTM/6DMG/20/t1-t3/5/5,5/0.3/mse/10.0/dot_product/"),
    ("6DMG", "results/linear_on_branch/tanh_in_LSTM/6DMG/20/t1-t3/5,5/5,5/0.3/mse/10.0/dot_product/"),
    ("6DMG", "results/linear_on_branch/tanh_in_Vanilla/6DMG/20/t1-t3/100,100/100,100/0.3/mse/10.0/dot_product"),
    ("6DMG", "results/linear_on_branch/tanh_in_Vanilla/6DMG/20/t1-t3/100/100,100/0.3/mse/10.0/dot_product"),
    ("UCIcharacter", "results/linear_on_branch/tanh_in_LSTM/UCIcharacter/20/t1-t3/20/20,20/0.3/mse/10.0/dot_product/"),
    ("UCIcharacter", "results/linear_on_branch/tanh_in_LSTM/UCIcharacter/20/t1-t3/10/10,10/0.3/mse/10.0/dot_product/"),
    ("UCIcharacter", "results/linear_on_branch/tanh_in_LSTM/UCIcharacter/20/t1-t3/5/5,5/0.3/mse/10.0/dot_product/"),
    ("UCIauslan", "results/linear_on_branch/tanh_in_LSTM/UCIauslan/10/100/100,100/0.3/mse/10.0/dot_product/"),
    ("UCIauslan", "results/linear_on_branch/tanh_in_LSTM/UCIauslan/10/10/10,10/0.3/mse/10.0/dot_product/"),
    ("UCIauslan", "results/linear_on_branch/tanh_in_LSTM/UCIauslan/10/5/5,5/0.3/mse/10.0/dot_product/"),
    ("UCIauslan", "results/linear_on_branch/tanh_in_LSTM/UCIauslan/10/30/30,30/0.3/mse/10.0/dot_product/")]


rnn_config_file_ = {}
rnn_config_file_["6DMG"] = "complete_matrix_rnn.json"
rnn_config_file_["UCIcharacter"] = "complete_matrix_rnn_UCIcharacter.json"
rnn_config_file_["UCIauslan"] = "complete_matrix_rnn_UCIauslan.json"
lsvm_config_file = "linear_svm.json"

def main():
    for dataset, model_dir in model_dirs:
        rnn_config_file = rnn_config_file_[dataset]

        # RNN and SVM
        rnn_conf_path = os.path.join(model_dir, rnn_config_file)
        fp = open(rnn_conf_path)
        loaded_data = json.load(fp)
        fp.close()
        loaded_data['params']['implementation'] = 1
        save_json(rnn_conf_path, loaded_data)
        command_rnn = 'CUDA_VISIBLE_DEVICES="" /usr/bin/python3 experiment/complete_matrix.py with ' + rnn_conf_path

        svm_conf_path = os.path.join(model_dir, "compute_classification_errors.json")
        command_svm = 'CUDA_VISIBLE_DEVICES="" /usr/bin/python3 experiment/compute_classification_errors.py with ' + svm_conf_path
                                                                                    
        os.system(command_rnn)
        os.system(command_svm)

        # Linear SVM
        rnn_conf_path = os.path.join(model_dir, rnn_config_file)
        fp = open(rnn_conf_path)
        loaded_data = json.load(fp)
        fp.close()
        loaded_data.pop("algorithm")
        loaded_data['params']['implementation'] = 1
        lsvm_conf_path = os.path.join(model_dir, lsvm_config_file)
        save_json(lsvm_conf_path, loaded_data)
        command_lsvm = 'CUDA_VISIBLE_DEVICES="" /usr/bin/python3 experiment/linear_svm.py with ' + lsvm_conf_path
        
        os.system(command_lsvm)






if __name__ == "__main__":
    main()


