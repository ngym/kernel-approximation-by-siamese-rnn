import sys, json, os, subprocess
from string import Template
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from datasets.k_fold_cross_validation_generators import KFold_UCIauslan, KFold, KFold_6DMGupperChar

if __name__ == "__main__":
    cnf_json_file = sys.argv[1]
    cnf = json.load(open(cnf_json_file, 'r'))

    program = cnf['env']['program']
    root_dir = cnf['env']['root_dir']
    python = cnf['env']['python']
    time = cnf['env']['time']
    num_thread = cnf['env']['num_thread']

    for dataset_type in list(cnf['dataset'].keys()):
        experiments_dir = os.path.join(root_dir, "USE_CASE_GAK_COMPLETION_1_VALIDATION")
        if dataset_type in {"UCIauslan", "UCItctodd"}:
            pkl_file_path = os.path.join(experiments_dir, "original_gram_files/gram_UCIauslan_sigma12.000.pkl")
            sample_dir = os.path.join(experiments_dir, "datasets/UCIauslan/all")
            kfold = KFold_UCIauslan
        elif dataset_type == "UCIcharacter":
            pkl_file_path = os.path.join(experiments_dir, "original_gram_files/gram_UCIcharacter_sigma20.000.pkl")
            sample_dir = os.path.join(experiments_dir, "datasets/UCIcharacter")
            kfold = KFold
        elif dataset_type in {"6DMGupperChar", "6DMG", "upperChar"}:
            pkl_file_path = os.path.join(experiments_dir, "original_gram_files/gram_upperChar_all_sigma20.000_t1-t3.pkl")
            sample_dir = os.path.join(experiments_dir, "datasets/6DMG_mat_112712/matR_char")
            kfold = KFold_6DMGupperChar
        else:
            raise ValueError("dataset must be one of UCIauslan, UCIcharacter or 6DMG")
        
        folds = kfold(pkl_file_path)
        
        cnf_d = cnf['dataset'][dataset_type]
        dataset_dir = os.path.join(experiments_dir, dataset_type)

        output_filename_format_ = cnf_d['output_filename_format']

        k = 0
        for fold in folds:
            if "data_attribute_type" in list(cnf_d.keys()): # 6DMG
                settings = [(dat, gak_sigma, os.path.join(dat, str(gak_sigma)),
                             Template(output_filename_format_).substitute(data_attribute_type=dat, gak_sigma=gak_sigma))
                            for dat in cnf_d['data_attribute_type']
                            for gak_sigma in cnf_d['gak_sigma']]
            else:
                settings = [(None, gak_sigma, os.path.join(str(gak_sigma)),
                             Template(output_filename_format_).substitute(gak_sigma=gak_sigma))
                            for gak_sigma in cnf_d['gak_sigma']]
                
            for dat, gak_sigma, ex_dir_, output_filename_format in settings:
                ex_dir = os.path.join(dataset_dir, ex_dir_, str(k))
                k += 1
                try:
                    os.makedirs(ex_dir)
                except FileExistsError:
                    pass            
                pkl_file = pkl_file_path.split("/")[-1]
                subprocess.run(["ln", "-s", pkl_file_path, ex_dir])
                gram_file = os.path.join(ex_dir, pkl_file)
                completionanalysisfile = gram_file.replace(".pkl", ".timelog")
                
                job_dict = dict(gram_file=gram_file,
                                sample_dir=sample_dir,
                                indices_to_drop=fold,
                                completionanalysisfile=os.path.join(ex_dir,
                                                                    completionanalysisfile),
                                gak_sigma=gak_sigma,
                                num_thread=num_thread,
                                output_dir=ex_dir,
                                output_filename_format=output_filename_format,
                                dataset_type=dataset_type)
    
                if 'data_mat_files' in list(cnf_d.keys()): # 6DMG
                    job_dict['data_mat_files'] = cnf_d['data_mat_files']
                    job_dict['data_attribute_type'] = dat
                elif 'data_tsd_files' in list(cnf_d.keys()): # UCIauslan
                    job_dict['data_tsd_files'] = cnf_d['data_tsd_files']
                elif 'data_mat_file' in list(cnf_d.keys()): # UCIcharacter
                    job_dict['data_mat_file'] = cnf_d['data_mat_file']
                else:
                    assert False
                
                json_file_name = os.path.join(ex_dir, output_filename_format + ".json")
                fd = open(json_file_name, "w")
                json.dump(job_dict, fd)
                fd.close()
                
                sh_file_name = os.path.join(ex_dir, output_filename_format + ".sh")
    
                time_file_name = os.path.join(ex_dir, output_filename_format + ".time")
                
                fd = open(sh_file_name, "w")
                fd.write(time + " -v -o " + time_file_name + " " +\
                         python + " " + program + " " + json_file_name + "\n")
                fd.close()
                
