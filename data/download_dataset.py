# Automatically does steps 2-4 of the readme

import os
import sys

sys.path.append("./audio")
sys.path.append("./video")

from audio_downloader import *
from audio_norm import *
from video_download import *
from MTCNN_detect import *
from frame_inspector import *

# ===== Settings =====

# Range of data to download from AVSpeech (excludes end_id(?))
start_id = 0
end_id = 1

# Whether to download from the training set or the test set
dl_from_training = True
normalise_data = True

# ====================


# Length of datasets
train_data_max_len = 2621845
test_data_max_len = 183273

def process_audio():
	# Download and trim audio wav files from youtube
	if dl_from_training:
		download_from_training()		
	else:
		download_from_testing()
	
	# Normalises audio data
	if normalise_data:
		norm(start_id, end_id)

def process_video():
	# # Download video data from online
	# download_video(start_idx=start_id, end_idx=end_id)
	
	# # Crop frames to fit face
	# mtcnn_detect(detect_range=(start_id,end_id))
	
	# Keep only valid frames
	frame_inspect(inspect_range=(start_id,end_id+1))

def main():
	# Download and process audio data from links
	os.chdir("./audio")
	# process_audio()
	
	# Download and process video data from links
	os.chdir("../video")
	process_video()

if __name__ == '__main__':
	main()

