

-----
# Siamese Recurrent Neural Network Acceleration for the Triangular Global Alignment Kernel

## Global Alignment Kernel
1. Copy compute_gram_matrix.json_sample to any name what you want.
        $cd experiments  
        $cp compute_gram_matrix.json_sample (new_json_file_name)
2. Specify
- dataset_type: UCIauslan, UCIcharacter or upperChar
- dataset_location: the directory which holds the dataset
- output_dir: directory to save the results
- output_filename_format: format of the output file
- sigma and triangular: gak variables
- data_augmentation_size: number of times to augment
3. Run the experiment
        $python3 compute_gram_matrix.py with (new_json_file_name)

## GRAM Matrix-based approach
### Matrix completion
1. Copy complete_matrix_(algorithm).json_sample to any name what you want.
        $cd experiments
        $cp compute_gram_matrix.json_sample (new_json_file_name)
2. Modify the parameters
        "fast_rnn" requires a pretrained model.
3. Run the experiment
        $python3 complete_matrix.py with (new_json_file_name)

### Classification error
1. Run the experiment
        $python3 compute_classification_errors.py with (parameter)=(new_value)
        
## Feature-based approach
### Feature mapping approximation
1. Copy linear_svm.json_sample to any name what you want.
        $cd experiments
        $cp linear_svm.json_sample (new_json_file_name)
2. Modify the parameters This method requires a pretrained model.
3. Run the experiment
        $python3 linear_svm.py with (new_json_file_name)



