# Automatically does steps 2-6* of the readme

import os
import sys

sys.path.append("./audio")
sys.path.append("./video")
sys.path.append("./AV_log")

from audio_downloader import download_from_training, download_from_testing
from audio_norm import norm
from video_download import download_video
from MTCNN_detect import mtcnn_detect
from frame_inspector import frame_inspect
from build_audio_database_v2 import build_database
from gentxtnew import main as gentxtnew

# ===== Settings =====

# Range of data to download from AVSpeech (excludes end_id(?))
start_id = 0
end_id = 301

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
		download_from_training(start_id, end_id)		
	else:
		download_from_testing(start_id, end_id)
	
	# Normalises audio data
	if normalise_data:
		norm(start_id, end_id)
	
	print("\n\n ===== Completed processing audio ===== \n")

def process_video():
	# Download video data from online
	try:
		download_video(start_idx=start_id, end_idx=end_id)
	except subprocess.CalledProcessError as e:
		print("Download failed -", e)
	
	# Crop frames to fit face
	mtcnn_detect(detect_range=(start_id,end_id))
	
	# Keep only valid frames
	frame_inspect(inspect_range=(start_id,end_id))
	
	print("\n\n ===== Completed processing video ===== \n")

def build_AV_databases():
	# TODO: Not sure what the exact condition is, but can fail without enough data
	assert end_id-start_id > 5, "Too few samples to generate database (probably...)"
	
	build_database(sample_range=(start_id,end_id))
	
	print("\n\n ===== Completed building databases ===== \n")

def main():
	# Download and process audio data from links
	os.chdir("./audio")
	# process_audio()
	
	# Download and process video data from links
	os.chdir("../video")
	# process_video()
	
	# Generate database from audio and visual data
	os.chdir("../audio")
	build_AV_databases()
	
	os.chdir("../AV_log")
	gentxtnew()
	
	# Then:
	# Run model/pretrain_model/pretrain_load_test.py
	# Rename face1022_emb to face_emb, and copy folder to data/video

if __name__ == '__main__':
	main()

