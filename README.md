

-----
# How to use.

## Global Alignment Kernel
1. Copy build_gak_experiment.json_sample to any name what you want.  
        $cd experiments  
        $cp build_gak_experiment.json_sample (new_json_file_name)
1. Specify
- "root directory" where the script create a new directory for experiments.
- the directory which holds the dataset you want to use in the file,
      - "data_mat_files" attribute for 6DMGupperChar, "data_tsd_files" attribute for UCIauslan and "data_mat_file" for UCIcharacter,
- the variable for GAK in this file, "gak_sigma" attribute.  
        $emacs (new_json_file_name)
1. Run the command to build directories for every experiment.  
        $python3 build_gak_experiment.py (new_json_file_name)  
Then you get structured directories such as "root directory"/GAK_EXPERIMENT/(dataset)/(gak_sigma). In each the deepest directory, you can find two files such as 
- gram_upperChar_all_sigma20.json
- gram_upperChar_all_sigma20.sh
1. Run each experiment.  
        $sh gram_upperChar_all_sigma20.sh


