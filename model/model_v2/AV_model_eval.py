## evaluate the model and generate the prediction
import sys
sys.path.append('../lib')
from keras.models import load_model
from model_ops import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
import utils
import tensorflow as tf
# super parameters
people_num = 2
NUM_GPU = 1

# PATH
# model_path = './saved_models_AO_with_norm/AOmodel-2p-015-0.02258.h5'
# model_path = './saved_AV_models/AVmodel-2p-092-0.70643.h5'	# trained on 21 data points
# model_path = './AVmodel-2p-002-0.87080.h5'	# 234K
model_path = './AVmodel-2p-006-0.75753.h5'	# 50K
dir_path = './pred/'
if not os.path.isdir(dir_path):
	os.mkdir(dir_path)

# database_path = '../../data/audio/audio_database/mix/'
database_path = '../../data/audio/AV_model_database/mix/'
face_path = '../../data/video/face_emb/'



# load data
testfiles = []
with open('../../data/AV_log/AVdataset_train.txt', 'r') as f:
	testfiles = f.readlines()

def parse_X_data(line,num_people=people_num,database_path=database_path,face_path=face_path):
	parts = line.split() # get each name of file for one testset
	mix_str = parts[0]
	name_list = mix_str.replace('.npy','')
	name_list = name_list.replace('mix-','',1)
	names = name_list.split('-')
	single_idxs = []
	for i in range(num_people):
		single_idxs.append(names[i])
	file_path = database_path + mix_str
	mix = np.load(file_path)
	face_embs = np.zeros((1,75,1,1792,num_people))
	for i in range(num_people):
		face_embs[0,:,:,:,i] = np.load(face_path+"%05d_face_emb.npy"%int(single_idxs[i]))

	return mix,single_idxs,face_embs

from model_loss import audio_discriminate_loss2 as audio_loss

# , custom_objects={'my_custom_func': my_custom_func}

# super parameters
people_num = 2
epochs = 100
initial_epoch = 0
batch_size = 2 # 4 to feed one 16G GPU
gamma_loss = 0.1
beta_loss = gamma_loss*2

loss_func = audio_loss(gamma=gamma_loss,beta=beta_loss, num_speaker=people_num)

import time
start_time = time.time()

# predict data
AV_model = load_model(model_path,custom_objects={"tf": tf, 'loss_func': loss_func})
if NUM_GPU > 1:
	parallel_model = ModelMGPU(AV_model,NUM_GPU)
	for line in testfiles:
		mix,single_idxs,face_embs = parse_X_data(line)
		mix_expand = np.expand_dims(mix, axis=0)
		cRMs = parallel_model.predict([mix_expand,face_embs])
		cRMs = cRMs[0]
		prefix = ""
		for idx in single_idxs:
			prefix += idx + "-"
		for i in range(len(cRMs)):
			cRM = cRMs[:,:,:,i]
			assert cRM.shape == (298,257,2)
			F = utils.fast_icRM(mix,cRM)
			T = utils.fast_istft(F,power=False)
			filename = dir_path+prefix+str(single_idxs[i])+'.wav'
			wavfile.write(filename,16000,T)


if NUM_GPU <= 1:
	for line in testfiles:
		t = time.time() - start_time
		print(line)
		print("time elapsed since start:", t)
		
		mix, single_idxs, face_embs = parse_X_data(line)
		mix_expand = np.expand_dims(mix, axis=0)
		cRMs = AV_model.predict([mix_expand, face_embs])
		cRMs = cRMs[0]
		prefix = ""
		for idx in single_idxs:
			prefix += idx + "-"
		for i in range(people_num):
			cRM = cRMs[:,:,:,i]
			assert cRM.shape == (298,257,2)
			F = utils.fast_icRM(mix,cRM)
			T = utils.fast_istft(F,power=False)
			filename = dir_path+prefix+single_idxs[i]+'.wav'
			wavfile.write(filename,16000,T)
















