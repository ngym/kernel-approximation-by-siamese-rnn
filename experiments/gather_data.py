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

baseline_dirs =[
    ("UCIcharacter", "si", "results/UCIcharacter/20/noaugmentation_softimpute/0.1/"),
    ("UCIcharacter", "gak", "results/UCIcharacter/20/gak/"),
    ("UCIcharacter", "si", "results/UCIcharacter/20/noaugmentation_softimpute/0.01/"),
    ("UCIcharacter", "knn", "results/UCIcharacter/20/noaugmentation_knn/0.1/"),
    ("UCIcharacter", "knn", "results/UCIcharacter/20/noaugmentation_knn/0.01/"),
    ("UCIauslan", "gak", "results/UCIauslan/10/gak/"),
    ("UCIauslan", "si", "results/UCIauslan/10/noaugmentation_softimpute/0.1"),
    ("UCIauslan", "si", "results/UCIauslan/10/noaugmentation_softimpute/0.01/"),
    ("UCIauslan", "knn", "results/UCIauslan/10/noaugmentation_knn/0.1/"),
    ("UCIauslan", "knn", "results/UCIauslan/10/noaugmentation_knn/0.01/"),
    ("6DMG", "gak", "results/6DMG/20/t1-t3/gak/"),
    ("6DMG", "si", "results/6DMG/20/t1-t3/noaugmentation_softimpute/0.1/"),
    ("6DMG", "si", "results/6DMG/20/t1-t3/noaugmentation_softimpute/0.01"),
    ("6DMG", "knn", "results/6DMG/20/t1-t3/noaugmentation_knn/0.1/"),
    ("6DMG", "knn", "results/6DMG/20/t1-t3/noaugmentation_knn/0.01/")]



rnn_svm_output_file = 'SiameseRnn_SVM_out.json'
lsvm_output_file = 'RnnBranch_SVM_out.json'
rnn_config_file_ = {}
rnn_config_file_["6DMG"] = "complete_matrix_rnn.json"
rnn_config_file_["UCIcharacter"] = "complete_matrix_rnn_UCIcharacter.json"
rnn_config_file_["UCIauslan"] = "complete_matrix_rnn_UCIauslan.json"
lsvm_config_file = "linear_svm.json"

output_file_ = {}
output_file_['gak'] = "gak_out.json"
output_file_['si'] = "si_out.json"
output_file_['knn'] = "knn_out.json"

def main():
    for dataset, model_dir in model_dirs:
        fp = open(os.path.join(model_dir, rnn_svm_output_file))
        rnn_svm = json.load(fp)
        fp.close()
        fp = open(os.path.join(model_dir, lsvm_output_file))
        lsvm = json.load(fp)
        fp.close()

        display = (model_dir,
                   # Siamese RNN and SVM
                   # as classification tool
                   rnn_svm['classification']['basics']['roc_auc'],
                   rnn_svm['classification']['basics']['f1'],
                   rnn_svm['classification']['each_seq']['virtual_classification_duration_per_calculated_sequence'],
                   rnn_svm['prediction']['each_seq']['virtual_completion_npsd_duration_per_calculated_sequence'],
                   rnn_svm['classification']['each_seq']['virtual_classification_duration_per_calculated_sequence'] + rnn_svm[
                       'prediction']['each_seq']['virtual_completion_npsd_duration_per_calculated_sequence'],
                   # as matrix completion tool
                   rnn_svm['prediction']['basics']['mean_absolute_error'],
                   rnn_svm['prediction']['each_elem']['virtual_completion_npsd_duration_per_calculated_element'],
                   rnn_svm['prediction']['each_elem']['elapsed_completion_npsd_duration_per_calculated_element'],
                   #
                   # Branch Siamese RNN and Linear SVM
                   # as classification tool
                   lsvm['classification']['basics']['roc_auc'],
                   lsvm['classification']['basics']['f1'],
                   lsvm['classification']['each_seq']['virtual_classification_duration_per_calculated_sequence'],
                   lsvm['prediction']['each_seq']['virtual_prediction_duration_per_calculated_sequence'],
                   lsvm['classification']['each_seq']['virtual_classification_duration_per_calculated_sequence'] + lsvm[
                       'prediction']['each_seq']['virtual_prediction_duration_per_calculated_sequence']
        )

        print("'%s' %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % display)

    for dataset, baseline_dir in baseline_dirs:
        output_file = output_file_[dataset]
        fp = open(os.path.join(baseline_dir, output_file))
        baseline = json.load(fp)
        fp.close()

        display = (baseline_dir,
                   # Siamese RNN and SVM
                   # as classification tool
                   rnn_svm['classification']['basics']['roc_auc'],
                   rnn_svm['classification']['basics']['f1'],
                   rnn_svm['classification']['each_seq']['virtual_classification_duration_per_calculated_sequence'],
                   rnn_svm['prediction']['each_seq']['virtual_completion_npsd_duration_per_calculated_sequence'],
                   rnn_svm['classification']['each_seq']['virtual_classification_duration_per_calculated_sequence'] + rnn_svm[
                       'prediction']['each_seq']['virtual_completion_npsd_duration_per_calculated_sequence'],
                   # as matrix completion tool
                   rnn_svm['prediction']['basics']['mean_absolute_error'],
                   rnn_svm['prediction']['each_elem']['virtual_completion_npsd_duration_per_calculated_element'],
                   rnn_svm['prediction']['each_elem']['elapsed_completion_npsd_duration_per_calculated_element']
        )

        print("'%s' %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f" % display)


        
if __name__ == "__main__":
    main()

        
