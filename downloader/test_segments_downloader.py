import unittest

import segments_downloader as SD
pwd = "/Users/ngym/Lorincz-Lab/programs/deep_kernel_learning/downloader/"
SD.output_directory = pwd
SD.youtube_dl_log_file = pwd + "youtube_dl_errored_YTID.log"
SD.ffmpeg_wav_log_file = pwd + "ffmpeg_wav_errored_YTID.log"
SD.ffmpeg_raw_log_file = pwd + "ffmpeg_raw_errored_YTID.log"
import csv

class TestYouTubeDownloader(unittest.TestCase):
    def setUp(self):
        SD.youtube_dl_logger = SD.Logger(SD.youtube_dl_log_file)
        SD.ffmpeg_wav_logger = SD.Logger(SD.ffmpeg_wav_log_file)
        SD.ffmpeg_raw_logger = SD.Logger(SD.ffmpeg_raw_log_file)
        
        #seg_meta = next(csv.reader(['--4gqARaEJE, 0.000, 10.000, "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"'], skipinitialspace=True))
        seg_meta = next(csv.reader(['x43sC-LNCZI, 0.000, 10.000, "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"'], skipinitialspace=True))
        self.ydl = SD.YouTubeDownloader(seg_meta, pwd)
    def test_download(self):
        data_type = "wav"
        self.ydl.download(data_type)
    def test_generate_audio_raw(self):
        data_type = "wav"
        self.ydl.download(data_type)
        self.ydl.generate_audio_raw()
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()    
