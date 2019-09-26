## evaluate the model and generate the prediction
import sys
sys.path.append('../lib')
from tensorflow.keras.models import load_model
import os
import scipy.io.wavfile as wavfile
import numpy as np
import tensorflow as tf
import utils

# super parameters
people_num = 2
NUM_GPU = 1

# PATH
# model_path = './AVmodel-2p-002-0.87080.h5'	# 234K
model_path = './AVmodel-2p-006-0.75753.h5'	# 50K


'''Load one entry of the dataset'''
mix_expand_path = "./test_inputs/" + "max_expand_00000-00011-.npy"
face_embs_path = "./test_inputs/" + "face_embs_00000-00011-.npy"
mix_path = "./test_inputs/" + "mix_00000-00011-.npy"

mix_expand = np.load(mix_expand_path)
face_embs = np.load(face_embs_path)
mix = np.load(mix_path)

'''Load model'''
# super parameters
people_num = 2
epochs = 100
initial_epoch = 0
batch_size = 2 # 4 to feed one 16G GPU
gamma_loss = 0.1
beta_loss = gamma_loss*2

from model_loss import audio_discriminate_loss2 as audio_loss

loss_func = audio_loss(gamma=gamma_loss,beta=beta_loss, num_speaker=people_num)
AV_model = load_model(model_path,custom_objects={"tf": tf, 'loss_func': loss_func})


'''Start testing'''

prefix = "--"
dir_path = './pred/'
if not os.path.isdir(dir_path):
	os.mkdir(dir_path)

import time

times = []
post_processing_times = []
num_tests = 30

for i in range(num_tests):
	'''Predict data'''
	start_time = time.time()
	cRMs = AV_model.predict([mix_expand, face_embs])
	end_time = time.time()
	runtime = end_time-start_time
	times.append(runtime)
	print("Time taken for run predict %d:"%(i+1), runtime)

	'''Save output as wav'''
	start_time = time.time()
	cRMs = cRMs[0]
	for j in range(people_num):
		cRM = cRMs[:,:,:,j]
		assert cRM.shape == (298,257,2)
		F = utils.fast_icRM(mix,cRM)
		T = utils.fast_istft(F,power=False)
		filename = dir_path+prefix+'.wav'
		wavfile.write(filename,16000,T)

	end_time = time.time()
	runtime = end_time-start_time
	post_processing_times.append(runtime)

	print("Time taken to post-process %d:"%(i+1), runtime)


list_of_time = post_processing_times

avg = sum(list_of_time) / len(list_of_time)
print("Number of test:", len(list_of_time))
print("Test times:", list_of_time)
print("Average running time:", avg)

# np.savetxt("Laptop_v2_postpro_times.csv", times, delimiter=",")
np.savetxt("Laptop_v2_postpro_times.csv", post_processing_times, delimiter=",")

'''Save output cRM in case we want to check'''
# np.save(face_embs_path.replace("face_embs", "cRMs_output"), cRMs)

# AV_model.save(model_path.replace("AVmodel", "AVmodel-resaved"))



# # for i in range(num_tests):
# '''Predict data'''
# start_time = time.time()
# cRMs = AV_model.predict([mix_expand, face_embs])
# end_time = time.time()
# runtime = end_time-start_time
# times.append(runtime)
# print("Time taken for run %d:"%(0+1), runtime)
# cRMs = cRMs[0]

# for i in range(num_tests):
	# start_time = time.time()
	# for j in range(people_num):
		# cRM = cRMs[:,:,:,j]
		# assert cRM.shape == (298,257,2)
		# F = utils.fast_icRM(mix,cRM)
		# T = utils.fast_istft(F,power=False)
		# filename = dir_path+prefix+'.wav'
		# wavfile.write(filename,16000,T)

	# end_time = time.time()
	# runtime = end_time-start_time
	# post_processing_times.append(runtime)

	# print("Time taken to post-process %d:"%(i+1), runtime)


