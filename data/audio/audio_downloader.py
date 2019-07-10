# Before running, make sure avspeech_train.csv and avspeech_test.csv are in catalog.
# if not, see the requirement.txt
# download and preprocess the data from AVspeech dataset
import sys

import os
if os.getcwd().rsplit("\\",1)[1] == "audio":
	sys.path.append("../lib")
else:
	sys.path.append("./lib")
import AVHandler as avh

import pandas as pd

# ===== Settings =====

# Range of data to download from AVSpeech
start_id = 0
end_id = 1

# Whether to download from the training set or the test set
dl_from_training = True

# ====================


# Length of datasets
train_data_max_len = 2621845
test_data_max_len = 183273

def m_link(youtube_id):
	# return the youtube actual link
	link = 'https://www.youtube.com/watch?v='+youtube_id
	return link

def m_audio(loc,name,cat,start_idx,end_idx):
	# make concatenated audio following by the catalog from AVSpeech
	# locals	| the location for file to store
	# name		| name for the wav mix file
	# cat		| the catalog with audio link and time
	# start_idx	| the starting index of the audio to download and concatenate
	# end_idx	| the ending index of the audio to download and concatenate

	for i in range(start_idx,end_idx):
		f_name = name+str(i)
		link = m_link(cat.loc[i,'link'])
		start_time = cat.loc[i,'start_time']
		end_time = start_time + 3.0
		avh.download(loc,f_name,link)
		avh.cut(loc,f_name,start_time,end_time)

def download_from_training(start_id=start_id, end_id=end_id):
	assert start_id >= 0 and end_id < train_data_max_len, "Data ID range is invalid"
	
	cat_train = pd.read_csv('catalog/avspeech_train.csv')

	# create 80000-90000 audios data from 290K
	avh.mkdir('audio_train')
	m_audio('audio_train', 'audio_train', cat_train, start_id, end_id)
	
def download_from_testing(start_id=start_id, end_id=end_id):
	assert start_id >= 0 and end_id < test_data_max_len, "Data ID range is invalid"
	
	cat_test = pd.read_csv('catalog/avspeech_test.csv')
	avh.mkdir('audio_test')
	m_audio('audio_test', 'audio_test', cat_test, start_id, end_id)

def download_audio():
	if dl_from_training:
		download_from_training()		
	else:
		download_from_testing()

if __name__ == '__main__':
	download_audio()

