import sys, json, subprocess

JSON_DIR = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/JSON/"
JOB_DIR = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/JOB/"

PYTHON = "/usr/bin/python3"
PROGRAM = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/gak_gram_matrix_completion.py"

if __name__ == "__main__":
    config_json_file = sys.argv[1]
    config_dict = json.load(open(config_json_file, 'r'))

    num_thread = config_dict['fixed']['num_thread']
    output_dir = config_dict['fixed']['output_dir']
    output_filename_format = config_dict['fixed']['output_filename_format']
    dataset_type = config_dict['fixed']['dataset_type']
    data_mat_files = config_dict['fixed']['data_mat_files']
    random_seed = config_dict['fixed']['random_seed']

    for data_attribute_type in config_dict['iterate']['data_attribute_type']:
        for gak_sigma in config_dict['iterate']['gak_sigma']:
            for incomplete_persentage in config_dict['iterate']['incomplete_persentage']:
                job_dict = dict(num_thread=num_thread,
                                outpout_dir=output_dir,
                                output_filename_format=output_filename_format,
                                dataset_type=dataset_type,
                                random_seeed=random_seed,
                                data_attribute_type=data_attribute_type,
                                gak_sigma=gak_sigma,
                                incomplete_persentage=incomplete_persentage,
                )
                output_filename_format_ = config_dict['fixed']['output_filename_format']
                output_filename_format_ = output_filename_format.replace("${dataset_type}",
                                                                         dataset_type)\
                                                                .replace("${data_attribute_type}",
                                                                         data_attribute_type)\
                                                                .replace("${gak_sigma}",
                                                                         ("%.3f" % gak_sigma))\
                                                                .replace("${incomplete_persentage}",
                                                                         str(incomplete_persentage))\
                                                                .replace("_${completion_alg}", "")
                json_file_name = JSON_DIR + output_filename_format_ + ".json"
                fd = open(json_file_name, "w")
                json.dump(job_dict, fd)
                fd.close()

                job_file_name = JOB_DIR + output_filename_format_ + ".job"
                fd = open(job_file_name, "w")
                fd.write(PYTHON + " " + PROGRAM + " " + json_file_name + "\n")
                fd.close()
                


