import glob

ROOT_DIR = "/home/ngym/NFSshare/Lorincz_Lab/fast-time-series-data-classification/gak_gram_matrix_completion/"
#ROOT_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/gak_gram_matrix_completion/"
JOB_DIR = ROOT_DIR + "JOB_COMPLETION2/"
TIME_DIR = ROOT_DIR + "TIME_COMPLETION/"
COMPLETION_ANALYSIS_DIR = ROOT_DIR + "COMPLETION_ANALYSIS/"

"""
COMPLETION_ALG = ["NuclearNormMinimization",
                  "SoftImpute",
                  "RNN"]
"""
COMPLETION_ALG = ["RNN"]

loss = [10, 20, 50]

PYTHON = "~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/python3"
TIME = "~/NFSshare/tflib/lib64/ld-2.17.so /usr/bin/time"

mats = []
mats += glob.glob(ROOT_DIR + "OUTPUT6DMG/*0.mat")
mats += glob.glob(ROOT_DIR + "OUTPUT_UCIcharacter/*0.mat")
mats += glob.glob(ROOT_DIR + "OUTPUT_UCItctodd/*0.mat")

for mat in mats:
    for CALG in COMPLETION_ALG:
        if CALG == "NuclearNormMinimization":
            py = ROOT_DIR + "matrix_completion_nuclearnormminimization.py"
        elif CALG == "SoftImpute":
            py = ROOT_DIR + "matrix_completion_softimpute.py"
        elif CALG == "RNN":
            py = ROOT_DIR + "matrix_completion_rnn.py"
        else:
            assert False
        for l in loss:
            output_filename_format = mat.split('/')[-1].replace(".mat", "_loss" + str(l) + "_" + CALG)
            job_file_name = JOB_DIR + output_filename_format + ".job"
            fd = open(job_file_name, "w")
            command = TIME + " -v -o " + TIME_DIR + output_filename_format + ".time " +\
                      PYTHON + " " + py + " " + mat + " " + str(l) + \
                      " " + COMPLETION_ANALYSIS_DIR + output_filename_format + ".error" + "\n"
            fd.write("echo $SHELL")
            fd.write("\n")
            fd.write("setenv LD_LIBRARY_PATH /home/ngym/NFSshare/tflib/lib64/:/home/ngym/NFSshare/tflib/usr/lib64/")
            fd.write("\n")
            fd.write(command)
            fd.write("\n")
            fd.close()
