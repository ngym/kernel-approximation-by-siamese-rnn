import os, sys, shutil, json
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.file_utils import save_json

model_dirs =[
    ("6DMG", "gak", "results/6DMG/20/t1-t3/gak/"),
    ("6DMG", "si", "results/6DMG/20/t1-t3/noaugmentation_softimpute/0.1/"),
    ("6DMG", "si", "results/6DMG/20/t1-t3/noaugmentation_softimpute/0.01"),
    ("6DMG", "knn", "results/6DMG/20/t1-t3/noaugmentation_knn/0.1/"),
    ("6DMG", "knn", "results/6DMG/20/t1-t3/noaugmentation_knn/0.01/"),
    ("UCIcharacter", "gak", "results/UCIcharacter/20/gak/"),
    ("UCIcharacter", "si", "results/UCIcharacter/20/noaugmentation_softimpute/0.1/"),
    ("UCIcharacter", "si", "results/UCIcharacter/20/noaugmentation_softimpute/0.01/"),
    ("UCIcharacter", "knn", "results/UCIcharacter/20/noaugmentation_knn/0.1/"),
    ("UCIcharacter", "knn", "results/UCIcharacter/20/noaugmentation_knn/0.01/"),
    ("UCIauslan", "gak", "results/UCIauslan/10/gak/"),
    ("UCIauslan", "si", "results/UCIauslan/10/noaugmentation_softimpute/0.1"),
    ("UCIauslan", "si", "results/UCIauslan/10/noaugmentation_softimpute/0.01/"),
    ("UCIauslan", "knn", "results/UCIauslan/10/noaugmentation_knn/0.1/"),
    ("UCIauslan", "knn", "results/UCIauslan/10/noaugmentation_knn/0.01/")]

output_file_ = {}
output_file_['knn'] = 'knn_out.json'
output_file_['si'] = 'si_out.json'
output_file_['gak'] = 'gak_out.json'

si_config_file_ = {}
si_config_file_["6DMG"] = "complete_matrix_si.json"
si_config_file_["UCIcharacter"] = "complete_matrix_si_UCIcharacter.json"
si_config_file_["UCIauslan"] = "complete_matrix_si_UCIauslan.json"
knn_config_file_ = {}
knn_config_file_["6DMG"] = "complete_matrix_knn.json"
knn_config_file_["UCIcharacter"] = "complete_matrix_knn_UCIcharacter.json"
knn_config_file_["UCIauslan"] = "complete_matrix_knn_UCIauslan.json"
gak_config_file_ = {}
gak_config_file_["6DMG"] = "complete_matrix_gak.json"
gak_config_file_["UCIcharacter"] = "complete_matrix_gak_UCIcharacter.json"
gak_config_file_["UCIauslan"] = "complete_matrix_gak_UCIauslan.json"

alg_={}
alg_['gak'] = gak_config_file_
alg_['si'] = si_config_file_
alg_['knn'] = knn_config_file_

lsvm_config_file = "linear_svm.json"

def main():
    for dataset, alg, model_dir in model_dirs:
        config_file = alg_[alg][dataset]
        output_file = output_file_[alg]
        
        # RNN and SVM
        conf_path = os.path.join(model_dir, config_file)
        fp = open(conf_path)
        loaded_data = json.load(fp)
        fp.close()

        loaded_data['output_file'] = output_file
        save_json(conf_path, loaded_data)
        command_mc = 'CUDA_VISIBLE_DEVICES="" /usr/bin/python3 experiments/complete_matrix.py with ' + conf_path
        
        svm_conf_path = os.path.join(model_dir, "compute_classification_errors.json")
        pohl = os.path.join(model_dir, loaded_data['output_filename_format'] + ".hdf5")
        of   = os.path.join(model_dir, output_file)
        dic = dict(pickle_or_hdf5_locations=[pohl],
                   output_location=of)
        save_json(svm_conf_path, dic)
        command_svm = 'CUDA_VISIBLE_DEVICES="" /usr/bin/python3 experiments/compute_classification_errors.py with ' + svm_conf_path

        
        os.system(command_mc)
        print("\n\n")
        os.system(command_svm)
        print("\n\n")


if __name__ == "__main__":
    main()


