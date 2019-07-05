# Automatically does steps 2-4 of the readme

import os
import sys

sys.path.append("./audio")
sys.path.append("./video")

from audio_downloader import *
from audio_norm import *
from video_download import *

# ===== Settings =====

# Range of data to download from AVSpeech
start_id = 0
end_id = 1

# Whether to download from the training set or the test set
dl_from_training = True
normalise_data = True

# ====================


# Length of datasets
train_data_max_len = 2621845
test_data_max_len = 183273

def main():
	os.chdir("./audio")

	# Download and trim audio wav files from youtube
	# if dl_from_training:
		# download_from_training()		
	# else:
		# download_from_testing()
	
	# # Normalises audio data
	# if normalise_data:
		# norm(start_id, end_id)
	
	os.chdir("../video")
	
	print(os.getcwd())
	# Download video data from online
	download_video()

if __name__ == '__main__':
	main()