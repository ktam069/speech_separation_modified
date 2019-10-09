import sys
import os
import datetime
import subprocess

sys.path.append("../../data/lib")
import AVHandler as avh

import pandas as pd


'''Video to download'''
video_id = 6


def download_video_frames(video_length=3.0):
	# loc		| the location for downloaded file
	
	loc=''
	path='../../data/audio/catalog/avspeech_train.csv'
	
	cat = pd.read_csv(path)

	avh.mkdir('frames')
	i = video_id
	
	try:
		command = 'cd %s&' % loc
		f_name = str(i)
		link = avh.m_link(cat.loc[i, 'link'])
		start_time = cat.loc[i, 'start_time']
		end_time = start_time + video_length
		start_time = datetime.timedelta(seconds=start_time)
		end_time = datetime.timedelta(seconds=end_time)
		
		# Run subprocess to get url
		n = subprocess.check_output(['youtube-dl', '-f', '”mp4“', '--get-url', str(link)])
		# Convert to string and remove whitespaces
		n = str(n.decode("utf-8")).split()[0]
		
		command += 'ffmpeg -i "%s" -c:v h264 -c:a copy -ss %s -to %s %s.mp4&' \
				   % (n,start_time, end_time, f_name)
		#ommand += 'ffmpeg -i %s.mp4 -r 25 %s.mp4;' % (f_name, 'clip_' + f_name)  # convert fps to 25

	except subprocess.CalledProcessError as e:
		print("Download failed -", e)
		return

	print(command)
	os.system(command)

if __name__ == '__main__':
	download_video_frames()
