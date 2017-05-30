import sys, json

PYTHON = "/usr/bin/python3"
PROGRAM ="/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/calculate_gram_matrix_by_gak.py"
TIME = "/usr/bin/time"

if __name__ == "__main__":
    config_json_file = sys.argv[1]
    config_dict = json.load(open(config_json_file, 'r'))

    json_dir = config_dict['env']['json_dir']
    job_dir = config_dict['env']['job_dir']
    time_dir = config_dict['env']['time_dir']
    
    num_thread = config_dict['fixed']['num_thread']
    output_dir = config_dict['fixed']['output_dir']
    output_filename_format = config_dict['fixed']['output_filename_format']
    dataset_type = config_dict['fixed']['dataset_type']
    data_mat_files = config_dict['fixed']['data_mat_files']

    for data_attribute_type in config_dict['iterate']['data_attribute_type']:
        for gak_sigma in config_dict['iterate']['gak_sigma']:
            job_dict = dict(num_thread=num_thread,
                            output_dir=output_dir,
                            output_filename_format=output_filename_format,
                            dataset_type=dataset_type,
                            data_mat_files=data_mat_files,
                            data_attribute_type=data_attribute_type,
                            gak_sigma=gak_sigma,
            )
            output_filename_format_ = config_dict['fixed']['output_filename_format']
            output_filename_format_ = output_filename_format.replace("${dataset_type}",
                                                                     dataset_type)\
                                                            .replace("${data_attribute_type}",
                                                                     data_attribute_type)\
                                                            .replace("${gak_sigma}",
                                                                     ("%.3f" % gak_sigma))
            json_file_name = json_dir + output_filename_format_ + ".json"
            fd = open(json_file_name, "w")
            json.dump(job_dict, fd)
            fd.close()
            
            job_file_name = job_dir + output_filename_format_ + ".job"
            fd = open(job_file_name, "w")
            fd.write(TIME + " -v -o " + time_dir + output_filename_format_ + " " +\
                     PYTHON + " " + PROGRAM + " " + json_file_name + "\n")
            fd.close()
            
