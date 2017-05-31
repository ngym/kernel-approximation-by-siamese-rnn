import sys, json

PYTHON = "/usr/bin/python3"
PROGRAM = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/audioset_full_gak_separately.py"
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

    for audioset_resampling_frequency in config_dict['iterate']['audioset_resampling_frequency']:
        for gak_sigma in config_dict['iterate']['gak_sigma']:
            for part in range(5):
                job_dict = dict(num_thread=num_thread,
                                output_dir=output_dir,
                                output_filename_format=output_filename_format,
                                dataset_type=dataset_type,
                                data_mat_files=data_mat_files,
                                audioset_resampling_frequency=audioset_resampling_frequency,
                                gak_sigma=gak_sigma,
                                part=part
                )
                output_filename_format_ = config_dict['fixed']['output_filename_format']
                output_filename_format_ = output_filename_format.replace("${dataset_type}",
                                                                         dataset_type)\
                                                                .replace("${audioset_resampling_frequency}",
                                                                         str(audioset_resampling_frequency))\
                                                                .replace("${gak_sigma}",
                                                                        ("%.3f" % gak_sigma))
                json_file_name = json_dir + output_filename_format_ + "_part" + str(part) + ".json"
                fd = open(json_file_name, "w")
                json.dump(job_dict, fd)
                fd.close()
                
                job_file_name = job_dir + output_filename_format_ + "_part" + str(part) + ".job"
                fd = open(job_file_name, "w")
                fd.write(TIME + " -v -o " + time_dir + output_filename_format_ + "_part" + str(part) + ".time " +\
                        PYTHON + " " + PROGRAM + " " + json_file_name + "\n")
                fd.close()
                


