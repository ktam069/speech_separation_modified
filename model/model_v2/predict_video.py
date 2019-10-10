'''
# ==========================================================
# Script for running a prediction on an input mp4 video
# (using a pretrained v2 AV model)
# ==========================================================
'''

import sys
sys.path.append('../lib')
from tensorflow.keras.models import load_model
import os
import scipy.io.wavfile as wavfile
import numpy as np
import tensorflow as tf
import librosa

from keras.models import Model
from mtcnn.mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt		# used for testing (visualising) purposes only

from keras.models import load_model as load_model_facenet
import matplotlib.image as mpimg

'''Note that the code from the utils script is needed (found under model/lib/)'''
import utils

'''
# ==========================================================
'''

# super parameters
num_people = 2

# PATH
VIDEO_NAME = "test_vid_6"
VIDEO_FILEPATH = "./"+VIDEO_NAME+".mp4"
MODEL_PATH = './AVmodel-2p-002-0.87080.h5'	# 234K
MODEL_PATH_FACENET = 'FaceNet_keras/facenet_keras.h5'

# Prediction output location
dir_path_pred = './pred/'
dir_path_frames = './frames/'
dir_path_faces = './face_input/'
dir_path_face_embs = './face_emb/'

'''
# ==========================================================
'''

def main(video_name=VIDEO_NAME):
	init(video_name)
	
	'''Generate a wav file for the audio stream'''
	mp4_to_wav(video_name, sr=16000)
	
	'''Generate face frames for the video stream'''
	generate_frames(video_name, fps=25)
	mtcnn_detect(video_name, video_length=3, fps=25)
	frame_inspect(video_name, video_length=3, fps=25)
	# TODO: use audio only model if no face is detected?
	
	'''Preprocess audio data'''
	preprocess_audio(video_name, sr=16000)
	
	'''Preprocess video data - FaceNet'''
	facenet_detect(video_name)
	
	'''Run prediction with AV model'''
	run_predict(video_name)


def init(video_name=VIDEO_NAME):
	if not os.path.isdir(dir_path_pred):
		os.mkdir(dir_path_pred)
	if not os.path.isdir(dir_path_frames):
		os.mkdir(dir_path_frames)
	if not os.path.isdir(dir_path_faces):
		os.mkdir(dir_path_faces)
	if not os.path.isdir(dir_path_face_embs):
		os.mkdir(dir_path_face_embs)
	
	'''
	# For convenience when testing
	'''
	import shutil
	if os.path.isdir(dir_path_frames):
		shutil.rmtree(dir_path_frames)
		os.mkdir(dir_path_frames)
	if os.path.isdir(dir_path_faces):
		shutil.rmtree(dir_path_faces)
		os.mkdir(dir_path_faces)
	if os.path.isdir(dir_path_face_embs):
		shutil.rmtree(dir_path_face_embs)
		os.mkdir(dir_path_face_embs)
	
	fname = '%s.wav'%video_name
	if os.path.isfile(fname):
		os.remove(fname)
	fname = 'o%s.wav'%video_name
	if os.path.isfile(fname):
		os.remove(fname)
	fname = 'preprocessed-%s.npy'%video_name
	if os.path.isfile(fname):
		os.remove(fname)
	'''
	# ============================
	'''
	

'''
# ==========================================================
# Separate audio into a wav file
# ==========================================================
'''

def mp4_to_wav(video_name=VIDEO_NAME, sr=16000, video_length=3):
	name = video_name
	
	# Convert the video to wav and resample
	command  = 'ffmpeg -i %s.mp4 o%s.wav&' % (name,name)
	command += 'ffmpeg -i o%s.wav -ar %d -ac 1 %s.wav&' % (name,sr,name)
	command += 'del o%s.wav' % name
	os.system(command)
	
	# Normalise audio (if it was done for the training data)
	if False:
		norm(video_name, sr)


def norm(video_name=VIDEO_NAME, sr=16000):
	path = '%s.wav'% video_name
	norm_path = 'norm_%s.wav'% video_name
	if (os.path.exists(path)):
		audio,_= librosa.load(path,sr=sr)
		max = np.max(np.abs(audio))
		norm_audio = np.divide(audio,max)
		wavfile.write(norm_path,sr,norm_audio)

'''
# ==========================================================
# Generate frames from an input video
# ==========================================================
'''

def generate_frames(video_name=VIDEO_NAME, fps=25):
	# command = 'ffmpeg -i %s.mp4 -y -f image2  -vframes 75 ./frames/%s-%%02d.jpg' % (video_name, video_name)
	command = 'ffmpeg -i %s.mp4 -vf fps=%d ./frames/%s-%%02d.jpg&' % (video_name, fps, video_name)
	os.system(command)

'''
# ==========================================================
# Crop frames to speaker's face
# ==========================================================
'''

tmp_face_x = [-1]*num_people
tmp_face_y = [-1]*num_people

def face_detect(file,detector):
	name = file.replace('.jpg', '').split('-')
	
	img = cv2.imread('%s%s'%(dir_path_frames,file))
	faces = detector.detect_faces(img)
	
	offset = 0
	
	for i in range(num_people):
		j = (i+offset) % num_people
		
		# check if detected faces
		if(len(faces)<=i or faces[i]['confidence']<0.6):
			print('Could not detect face for speaker no. %d: %s'%(i+1,file))
			return
		
		bounding_box = faces[j]['box']
		'''Dealing with permutation problem of faces'''
		while tmp_face_x[i] >= 0 and (abs(tmp_face_x[i] - bounding_box[0]) > 100 or abs(tmp_face_y[i] - bounding_box[1]) > 100):
			offset += 1
			if offset >= num_people:
				break
			j = (i+offset) % num_people
			bounding_box = faces[j]['box']
		
		bounding_box[1] = max(0, bounding_box[1])
		bounding_box[0] = max(0, bounding_box[0])
		print(file," ",bounding_box)
		
		crop_img = img[bounding_box[1]:bounding_box[1] + bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]
		crop_img = cv2.resize(crop_img,(160,160))
		cv2.imwrite('%s/frame_'%dir_path_faces + name[0] + '_p%d_'%i + name[1] + '.jpg', crop_img)
		# crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
		# plt.imshow(crop_img)
		# plt.show()
		
		'''Dealing with permutation problem of faces'''
		tmp_face_x[i] = bounding_box[0]
		tmp_face_y[i] = bounding_box[1]
		
		'''Write images with faces outlined - for demo purposes'''
		cv2.rectangle(img,
			(bounding_box[0], bounding_box[1]),
			(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
			(0,155,255),
			2)
	cv2.imwrite('%s/demo_frame_'%dir_path_frames + name[0] + '_' + name[1] + '.jpg', img)


def mtcnn_detect(video_name, video_length=3, fps=25):
	detector = MTCNN()
	for j in range(1,video_length*fps+1):
		file_name = "%s-%02d.jpg"%(video_name, j)
		if (not os.path.exists('%s%s' % (dir_path_frames, file_name))):
			print('cannot find input: ' + '%s%s' % (dir_path_frames, file_name))
			continue
		face_detect(file_name, detector)

'''
# ==========================================================
# Check that the video face inputs are valid
# ==========================================================
'''

def check_frame(name,part,dir=dir_path_faces):
	path = dir + "/frame_%s_%02d.jpg"%(name,part)
	print(path)
	if(not os.path.exists(path)): return False
	return True

def frame_inspect(video_name=VIDEO_NAME, video_length=3, fps=25):
	i = video_name
	valid = True
	print('processing frames for video %s'%i)
	for j in range(1,video_length*fps+1):
		if(check_frame(i,j)==False):
			valid = False
			print('At least one frame for video %s is not valid'%i)
			break
	return valid

'''
# ==========================================================
# Use FaceNet to generate face embeddings
# ==========================================================
'''

# TODO: Not yet adapted to work with varying length videos

def facenet_detect(video_name=VIDEO_NAME):
	FACE_INPUT_PATH = dir_path_faces
	
	print("Running FaceNet...")
	print("Using frames from:", FACE_INPUT_PATH)

	model = load_model_facenet(MODEL_PATH_FACENET)
	model.summary()
	avgPool_layer_model = Model(inputs=model.input,outputs=model.get_layer('AvgPool').output)
	# print(avgPool_layer_model.predict(data))

	for j in range(num_people):
		try:
			line = "frame_"+video_name+'_p%d'%j
			
			embtmp = np.zeros((75, 1, 1792))
			headname = line.strip()
			tailname = ''
			for i in range(1, 76):
				if i < 10:
					tailname = '_0{}.jpg'.format(i)
				else:
					tailname = '_' + str(i) + '.jpg'
				picname = headname + tailname
				# print(picname)
				I = mpimg.imread(FACE_INPUT_PATH + picname)
				I_np = np.array(I)
				I_np = I_np[np.newaxis, :, :, :]
				# print(I_np.shape)
				# print(avgPool_layer_model.predict(I_np).shape)
				embtmp[i - 1, :] = avgPool_layer_model.predict(I_np)

			# print(embtmp.shape)
			# people_index = int(line.strip().split('_')[1])
			# npname = '{:05d}_face_emb.npy'.format(people_index)
			npname = '%s_face_emb_p%d.npy'%(video_name,j)
			print(npname)

			np.save(dir_path_face_embs + npname, embtmp)
			with open('faceemb_dataset.txt', 'a') as d:
				d.write(npname + '\n')
		except Exception as e:
			print('No face input for speaker', j, "\n", e)
	
	print("Finished running FaceNet...")


'''
# ==========================================================
# Preprocess data
# ==========================================================
'''

def preprocess_audio(video_name=VIDEO_NAME, sr=16000):
	path = "%s.wav"%video_name
	
	data, _ = librosa.load(path, sr=sr)
	data = utils.fast_stft(data)
	
	name = 'preprocessed-%s'%video_name
	np.save('%s.npy'%name,data)

'''
# ==========================================================
# Custom loss function
# ==========================================================
'''

import keras.backend as K

def audio_discriminate_loss2(gamma=0.1,beta = 2*0.1,num_speaker=2):
	def loss_func(S_true,S_pred,gamma=gamma,beta=beta,num_speaker=num_speaker):
		sum_mtr = K.zeros_like(S_true[:,:,:,:,0])
		for i in range(num_speaker):
			sum_mtr += K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,i])
			for j in range(num_speaker):
				if i != j:
					sum_mtr -= gamma*(K.square(S_true[:,:,:,:,i]-S_pred[:,:,:,:,j]))

		for i in range(num_speaker):
			for j in range(i+1,num_speaker):
				#sum_mtr -= beta*K.square(S_pred[:,:,:,i]-S_pred[:,:,:,j])
				#sum_mtr += beta*K.square(S_true[:,:,:,:,i]-S_true[:,:,:,:,j])
				pass
		#sum = K.sum(K.maximum(K.flatten(sum_mtr),0))

		loss = K.mean(K.flatten(sum_mtr))

		return loss
	return loss_func

audio_loss = audio_discriminate_loss2

# super parameters
gamma_loss = 0.1
beta_loss = gamma_loss*2

'''
# ==========================================================
# Predict video
# ==========================================================
'''

def run_predict(video_name=VIDEO_NAME):
	
	'''Load audio data'''
	audio_data = np.load('preprocessed-%s.npy'%video_name)
	print(audio_data.shape)
	# TODO: check shape - first dim should be 298
	audio_data = audio_data[:298]
	if len(audio_data) < 298:
		a = np.zeros((298,257,2))
		a[:len(audio_data),:,:] = audio_data
		audio_data = a
	print(audio_data.shape)
	mix_expand = np.expand_dims(audio_data, axis=0)
	print(mix_expand.shape)
	
	'''Load visual data'''
	face_embs = np.zeros((1,75,1,1792,num_people))
	print(face_embs.shape)
	for i in range(num_people):
		try:
			# face_embs[1,:,:,:,i] = np.load(dir_path_face_embs+"%s_face_emb.npy"%single_idxs[i])
			'''Currently does not use the correct face input for both speakers (uses the same images for both right now)'''
			face_embs[0,:,:,:,i] = np.load(dir_path_face_embs+"%s_face_emb_p%d.npy"%(video_name,i))
		except Exception as e:
			print('No face embedding for speaker', i, "\n", e)
	'''TODO: use Google Vision AI to find the face embedding for each speaker'''
	
	
	# '''Load pretrained model'''
	loss_func = audio_loss(gamma=gamma_loss,beta=beta_loss, num_speaker=num_people)
	AV_model = load_model(MODEL_PATH,custom_objects={"tf": tf, 'loss_func': loss_func})

	# '''Predict data'''
	cRMs = AV_model.predict([mix_expand, face_embs])
	cRMs = cRMs[0]

	# '''Save output as wav'''
	for j in range(num_people):
		cRM = cRMs[:,:,:,j]
		assert cRM.shape == (298,257,2)
		F = utils.fast_icRM(audio_data,cRM)
		T = utils.fast_istft(F,power=False)
		filename = dir_path_pred+'pred_%s_output%d.wav'%(video_name,j)
		wavfile.write(filename,16000,T)


'''
# ==========================================================
# Test predicting on a video
# ==========================================================
'''

if __name__ == '__main__':
	main(VIDEO_NAME)

