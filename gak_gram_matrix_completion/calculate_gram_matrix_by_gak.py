import numpy as np
from collections import OrderedDict

import scipy as sp
from scipy import io
from scipy.io import wavfile
from scipy import signal

import subprocess, functools, sys, threading, glob, json, csv
import concurrent.futures

from python_speech_features import mfcc
from python_speech_features import logfbank

from plot_gram_matrix import plot

from string import Template

from gak import gak

class Logger:
    def __init__(self, log_file):
        self.__lock = threading.Lock()
        self.__fd = open(log_file, 'w')
    def write(self, msg):
        self.__lock.acquire()
        try:
            self.__fd.write(msg)
            #self.__fd.flush()
        finally:
            self.__lock.release()
    def __del__(self):
        self.__fd.write("\n")
        self.__fd.close()

gak_logger = None

class GRAMmatrix:
    def __init__(self, seq_ids):
        self.__lock = threading.Lock()
        self.gram = OrderedDict()
        for seq_id1 in seq_ids:
            self.gram[seq_id1] = OrderedDict()
            for seq_id2 in seq_ids:
                self.gram[seq_id1][seq_id2] = -1
    def register(self, seq_id1, seq_id2, value):
        self.__lock.acquire()
        try:
            self.gram[seq_id1][seq_id2] = value
            self.gram[seq_id2][seq_id1] = value
        finally:
            self.__lock.release()

gram = None

seqs = {}

def read_and_resample_worker(f, frequency):
    mat_filename = f.replace(".wav", ("_freq" + str(frequency) + ".mat"))
    print(mat_filename)
    try:
        mat = io.loadmat(mat_filename)
        resampled_data = mat['resampled_data']
        print("read from mat")
    except:
        rate, data = io.wavfile.read(f)
        data = logfbank(data, rate)
        length = frequency * data.__len__() // rate
        print("read from wav")
        resampled_data = signal.resample(data, length)
    return resampled_data

def audioset_read_wavs_and_build_seqs(files, audioset_resampling_frequency):
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        future_to_file = {executor.submit(read_and_resample_worker, f, audioset_resampling_frequency): f
                           for f in files}
        print("reading %d files." % future_to_file.__len__())
        for future in concurrent.futures.as_completed(future_to_file):
            f = future_to_file[future]
            resampled_data = future.result()
            if resampled_data.ndim > 1:
                # stereo
                # convert stereo to mono
                # https://librosa.github.io/librosa/_modules/librosa/core/audio.html          
                seqs[f] = np.array([[np.mean([np.float64(elem) for elem in sample], axis=0)] for sample in resampled_data])
            else:
                # mono
                seqs[f] = np.array([[np.float64(sample)] for sample in resampled_data])
            #print(seqs[f])
            print(str(seqs.__len__()) + " " + f)
            sys.stdout.flush()

def SixDMG_read_mats_and_build_seqs(files, attribute_type):
    for f in files:
        mat = io.loadmat(f)
        seqs[f] = np.array(SixDMG_pick_attribute(mat['gest'].transpose(), attribute_type)).astype('float64')

def SixDMG_pick_attribute(ll, attribute_type):
    retval = []
    if attribute_type == "position":
        for l in ll:
            retval.append(l[1:4])
    elif attribute_type == "velocity":
        for i in range(ll.__len__() - 1):
            retval.append([(ll[i+1][1] - ll[i][1])/((ll[i+1][0] - ll[i][0])/1000),
                           (ll[i+1][2] - ll[i][2])/((ll[i+1][0] - ll[i][0])/1000),
                           (ll[i+1][3] - ll[i][3])/((ll[i+1][0] - ll[i][0])/1000)])
    elif attribute_type == "acceleration":
        for l in ll:
            retval.append(l[8:11])
    elif attribute_type == "angularvelocity":
        for l in ll:
            retval.append(l[11:14])
    elif attribute_type == "orientation":
        for l in ll:
            retval.append(l[4:8])
    elif attribute_type == "all":
        for l in ll:
            retval.append(l[1:14])
    else:
        print("attribute type error.")
        assert False
    return retval

def UCIcharacter_read_mat_and_build_seqs(mat_filename):
    mat = io.loadmat(mat_filename)
    displayname = [k[0] for k in mat['consts']['key'][0][0][0]]
    classes = mat['consts'][0][0][4][0]
    labels = []
    for c in classes:
        labels.append(displayname[c-1])
    i = 0
    for l in labels:
        seqs[l + str(i)] = mat['mixout'][0][i].T
        i += 1
    gram = GRAMmatrix(sorted(seqs.keys()))

def UCItctodd_read_tsd_and_build_seqs(files):
    for f in files:
        reader = csv.reader(open(f, "r"), delimiter='\t')
        seq = []
        for r in reader:
            seq.append(r)
        seqs[f] = np.float64(np.array(seq))
    
def worker_for_f1(files, f1index, f2indices, gak_sigma, triangular):
    f1 = files[f1index]
    seq1 = seqs[f1]
    ret_dict = {}
    for f2index in f2indices:
        f2 = files[f2index]
        seq2 = seqs[f2]
        ret_dict[f2] = gak(seq1, seq2, gak_sigma, triangular)
    return ret_dict
    
def main():
    config_json_file = sys.argv[1]
    config_dict = json.load(open(config_json_file, 'r'))
    
    num_thread = config_dict['num_thread']

    dataset_type = config_dict['dataset_type']

    gak_sigma = np.float64(config_dict['gak_sigma'])

    output_dir = config_dict['output_dir']
    
    if dataset_type in {"num", "upperChar", "6DMGupperChar"}:
        # 6DMG
        data_files = config_dict['data_mat_files']
        data_attribute_type = config_dict['data_attribute_type']
        output_filename_format = Template(config_dict['output_filename_format']).safe_substitute(
            dict(dataset_type=dataset_type,
                 data_attribute_type=data_attribute_type,
                 gak_sigma=("%.3f" % gak_sigma)))
        files = []
        for df in data_files:
            files_ = glob.glob(df)
            print(files_[:3])
            print("...")
            print(files_[-3:])
            files += files_
        files = sorted(files)
    elif dataset_type == "audioset":
        # audioset
        data_files = config_dict['data_mat_files']
        audioset_resampling_frequency = config_dict['audioset_resampling_frequency']
        output_filename_format = Template(config_dict['output_filename_format']).safe_substitute(
            dict(dataset_type=dataset_type,
                 audioset_resampling_frequency=audioset_resampling_frequency,
                 gak_sigma=("%.3f" % gak_sigma)))
        files = []
        for df in data_files:
            files_ = glob.glob(df)
            print(files_[:3])
            print("...")
            print(files_[-3:])
            files += files_
        files = sorted(files)
    elif dataset_type == "UCIcharacter":
        data_file = config_dict['data_mat_file']
        output_filename_format = Template(config_dict['output_filename_format']).safe_substitute(
            dict(dataset_type=dataset_type,
                 gak_sigma=("%.3f" % gak_sigma)))
    elif dataset_type == "UCIauslan":
        data_files = config_dict['data_tsd_files']
        output_filename_format = Template(config_dict['output_filename_format']).safe_substitute(
             dict(dataset_type=dataset_type,
                 gak_sigma=("%.3f" % gak_sigma)))
        files = []
        for df in data_files:
            files_ = glob.glob(df)
            print(files_[:3])
            print("...")
            print(files_[-3:])
            files += files_
        files = sorted(files)
    else:
        assert False

    html_out_full_gak = output_dir + output_filename_format.replace("${completion_alg}", "FullGAK") + ".html" 
    mat_out_full_gak = output_dir + output_filename_format.replace("${completion_alg}", "FullGAK") + ".mat" 
    
    gak_logfile = output_dir + output_filename_format.replace("_${completion_alg}", "") + ".log"
    gak_logger = Logger(gak_logfile)

    if dataset_type in {"num", "upperChar", "6DMGupperChar"}:
        # 6DMG
        SixDMG_read_mats_and_build_seqs(files, data_attribute_type)
    elif dataset_type == "audioset":
        # audioset
        audioset_read_wavs_and_build_seqs(files, audioset_resampling_frequency)
    elif dataset_type == "UCIcharacter":
        UCIcharacter_read_mat_and_build_seqs(data_file)
        files = sorted(seqs.keys())
    elif dataset_type == "UCItctodd":
        UCItctodd_read_tsd_and_build_seqs(files)
    else:
        assert False
    
    gram = GRAMmatrix(files)

    similarities = []
    file_num = files.__len__()

    gak_triangular = np.median([seq.__len__() for seq in seqs.values()]) * 0.5
    print("gak_triangular (median of sequences): " + repr(gak_triangular))
    print("average length of sequences: " + repr(np.mean([seq.__len__() for seq in seqs.values()])))

    futures = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_thread) as executor:
        print("Start submitting jobs.")
        future_to_files = {executor.submit(worker_for_f1, files, f1index,
                                           range(f1index, file_num), gak_sigma, gak_triangular):
                           files[f1index]
                           for f1index in range(file_num)}
        num_futures = future_to_files.__len__()
        print(str(num_futures) + " jobs are submitted.")
        num_finished_jobs = 0
        for future in concurrent.futures.as_completed(future_to_files):
            f1 = future_to_files[future]
            ret_dict = future.result()
            ret_dict_keys = list(ret_dict.keys())
            for f2 in ret_dict_keys:
                value = ret_dict[f2]
                gram.register(f1, f2, value)
                #gak_logger.write(f1 + ", " + f2 + ", " + str(value) + "\n")
            num_finished_jobs += 1
            print(str(num_finished_jobs) + "/" + str(num_futures), end=" ")
            sys.stdout.flush()

    print(" ")

    similarities = []
    for i in gram.gram.values():
        similarities.append(list(i.values()))

    # "FullGAK"
    plot(html_out_full_gak,
         similarities, files)
    io.savemat(mat_out_full_gak, dict(gram=similarities, indices=files))
    print("FullGAK files are output.")

if __name__ == "__main__":
    main()
