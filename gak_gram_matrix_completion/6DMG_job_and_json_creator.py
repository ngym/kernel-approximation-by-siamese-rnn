import sys, json, os
from string import Template

if __name__ == "__main__":
    cnf_json_file = sys.argv[1]
    cnf = json.load(open(cnf_json_file, 'r'))

    program = cnf['env']['program']
    root_dir = cnf['env']['root_dir']
    python = cnf['env']['python']
    time = cnf['env']['time']
    num_thread = cnf['env']['num_thread']

    for dataset_type in list(cnf['dataset'].keys()):
        cnf_d = cnf['dataset'][dataset_type]
        dataset_dir = os.path.join(root_dir, "GAK_EXPERIMENT", dataset_type)

        output_filename_format_ = cnf_d['output_filename_format']

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
            ex_dir = os.path.join(dataset_dir, ex_dir_)
            try:
                os.makedirs(ex_dir)
            except FileExistsError:
                pass            
            job_dict = dict(num_thread=num_thread,
                            output_dir=ex_dir,
                            output_filename_format=output_filename_format,
                            dataset_type=dataset_type,
                            gak_sigma=gak_sigma)

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

            time_file_name = os.path.join(ex_dir, output_filename_format + ".time ")
            
            fd = open(sh_file_name, "w")
            fd.write(time + " -v -o " + time_file_name+\
                     python + " " + program + " " + json_file_name + "\n")
            fd.close()
            
