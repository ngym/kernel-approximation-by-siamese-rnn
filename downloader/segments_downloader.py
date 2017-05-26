#!/usr/local/bin/python3

"""
1. Pick a YouTube video which has the input label.
2. Find the URL of the audio file of the video
3. Download the audio file as .wav format.
4. Convert the .wav into raw pulse code modulation u32le (unsigned 32 bit little endian).
5. Repeat 1-4 until all videos the label is annotated are processed.
"""

import sys, csv, subprocess, threading
import concurrent.futures

ROOT_DIR = "/Users/ngym/Lorincz-Lab/project/fast_time-series_data_classification/program/downloader/"
OUTPUT_DIRECTORY = ROOT_DIR + "downloads/"
CLASS_LABELS_INDICES_FILE = ROOT_DIR + "class_labels_indices.csv"
YOUTUBE_DL_LOG_FILE = ROOT_DIR + "youtube_dl_errored_YTID.log"
FFMPEG_DL_LOG_FILE = ROOT_DIR + "ffmpeg_wav_errored_YTID.log"
FFMPEG_RAW_LOG_FILE = ROOT_DIR + "ffmpeg_raw_errored_YTID.log"

class Logger:
    def __init__(self, log_file):
        self.__lock = threading.Lock()
        self.__fd = open(log_file, 'w')
    def write(self, msg):
        self.__lock.acquire()
        try:
            self.__fd.write(msg)
            self.__fd.flush()
        finally:
            self.__lock.release()
    def __del__(self):
        self.__fd.write("\n")
        self.__fd.close()

youtube_dl_logger = None
ffmpeg_wav_logger = None
ffmpeg_raw_logger = None

class TableIndexDisplaynameMID:
    """
    MID is Knowledge Graph Machine ID. See the paper by Gemmeke et al. for details.
    """
    def __init__(self):
        table_reader = csv.reader(open(CLASS_LABELS_INDICES_FILE, 'r'))
        self.index_to_mid = {}
        self.index_to_displayname = {}
        self.mid_to_displayname = {}
        self.mid_to_index = {}
        self.displayname_to_index = {}
        self.displayname_to_mid = {}
        for entry in table_reader:
            index, mid, displayname = entry
            self.index_to_mid[index] = mid
            self.index_to_displayname[index] = displayname
            self.mid_to_displayname[mid] = displayname
            self.mid_to_index[mid] = index
            self.displayname_to_index[displayname] = index
            self.displayname_to_mid[displayname] = mid

class YouTubeDownloader():
    def __init__(self, seg_meta, out_dir, displayname):
        self.out_dir = out_dir
        self.downloaded_file = None
        self.meta_data = seg_meta
        self.YTID, self.start, self.end, self.positive_labels = seg_meta
        self.displayname = displayname
    def download_audio(self, data_type):
        audio_url = self.__get_audio_url()
        if audio_url is None:
            return
        if self.__download(audio_url, data_type) != 0:
            return
    def __get_audio_url(self):
        for i in range(10):
            youtube_cmd = ["youtube-dl",
                           "-x",
                           "-g",
                           "https://www.youtube.com/watch?v=" + self.YTID]
            #print(youtube_cmd)
            try:
                audio_url_raw = subprocess.check_output(youtube_cmd, universal_newlines=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as cpe:
                error_cause = cpe.output
            except:
                import traceback
                traceback.print_exc()
            else:
                audio_url = audio_url_raw.replace('\n', '')
                retval = audio_url
                return retval
        youtube_dl_logger.write(self.YTID + ", " + error_cause)
        print("failed youtube-dl:" + self.YTID)
        return None
    def __download(self, url, data_type):
        dl_cmd = ["ffmpeg",
                  "-i", url,
                  "-ss", self.start,
                  "-to", self.end,
                  "-loglevel", "quiet",
                  self.out_dir + self.displayname + "_YTID" + self.YTID + "." + data_type]
        #print(dl_cmd)
        for i in range(100):
            try:
                subprocess.check_output(dl_cmd, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as cpe:
                error_cause = cpe.output
            except:
                import traceback
                traceback.print_exc()
            else:
                self.downloaded_file = self.out_dir + self.displayname + "_YTID" + self.YTID + "." + data_type
                return 0
        ffmpeg_wav_logger.write(self.YTID + ", " + error_cause)
        print("failed downloading " + repr(data_type) + ":" + self.YTID)
        return -1
    def generate_audio_raw(self):
        #print(self.downloaded_file)
        if self.downloaded_file is None:
            #print("Could not start generating raw:" + self.YTID)
            return -1
        raw_cmd = ["ffmpeg",
                   "-i", self.downloaded_file,
                   "-loglevel", "quiet",
                   "-f", "u16le",
                   "-acodec", "pcm_u16le",
                   self.out_dir + self.displayname + "_YTID" + self.YTID + ".raw"]
        for i in range(100):
            try:
                subprocess.check_output(raw_cmd)
            except subprocess.CalledProcessError:
                pass
            except:
                import traceback
                traceback.print_exc()
            else:
                return 0
        ffmpeg_raw_logger.write(self.YTID)
        return -1

def worker(meta_data, data_type, out_dir, displayname):
    dloader = YouTubeDownloader(meta_data, out_dir, displayname)
    dloader.download_audio(data_type)
    dloader.generate_audio_raw()
    print("work finish: " + repr(meta_data))
        
if __name__ == "__main__":
    # python3 segments_downloader.py ../../dataset/audioset/dataset_split_csv/eval_segments.csv 2 wav Bark
    filename = sys.argv[1]
    num_thread = int(sys.argv[2])
    data_type = sys.argv[3]
    displayname = label = sys.argv[4]
    
    youtube_dl_logger = Logger(YOUTUBE_DL_LOG_FILE)
    ffmpeg_wav_logger = Logger(FFMPEG_DL_LOG_FILE)
    ffmpeg_raw_logger = Logger(FFMPEG_RAW_LOG_FILE)
    
    table = TableIndexDisplaynameMID()
    mid = table.displayname_to_mid[displayname]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        meta_data_reader = csv.reader(open(filename, 'r'), skipinitialspace=True)
        for meta_data in meta_data_reader:
            YTID, start, end, positive_MIDs_raw = meta_data
            positive_MIDs = next(csv.reader([positive_MIDs_raw]))
            if mid in positive_MIDs:
                executor.submit(worker, meta_data, data_type, OUTPUT_DIRECTORY, displayname)

#a=`youtube-dl -x -g "https://www.youtube.com/watch?v=-1iKLvsRBbE"`; ffmpeg -i "$a" -ss 10 -to 20 -f u16le -acodec pcm_u16le asdf.raw    
