import sys
sys.path.append('../lib')
import model_AV_new as AV
from model_ops import ModelMGPU,latest_file
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Model, load_model
from MyGenerator import AVGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import os
from model_loss import audio_discriminate_loss2 as audio_loss
import tensorflow as tf

# create AV model
#############################################################
RESTORE = True
# If set true, continue training from last checkpoint
# needed change 1:h5 file name, 2:epochs num, 3:initial_epoch

# super parameters
people_num = 2
epochs = 100
initial_epoch = 20
batch_size = 2 # 4 to feed one 16G GPU
gamma_loss = 0.1
beta_loss = gamma_loss*2

# physical devices option to accelerate training process
workers = 1 # num of core
use_multiprocessing = False
NUM_GPU = 1

# PATH
path = './saved_AV_models' # model path
database_dir_path = '../../data/'
#############################################################

# create folder to save models
folder = os.path.exists(path)
if not folder:
	os.makedirs(path)
	print('create folder to save models')
filepath = path + "/AVmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.5f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


#############################################################
# automatically change lr
def scheduler(epoch):
	ini_lr = 0.00001
	lr = ini_lr
	if epoch >= 5:
		lr = ini_lr / 5
	if epoch >= 10:
		lr = ini_lr / 10
	return lr

rlr = LearningRateScheduler(scheduler, verbose=1)
#############################################################
# read train and val file name
# format: mix.npy single.npy single.npy
trainfile = []
valfile = []
with open((database_dir_path+'AV_log/AVdataset_train.txt'), 'r') as t:
	trainfile = t.readlines()
with open((database_dir_path+'AV_log/AVdataset_val.txt'), 'r') as v:
	valfile = v.readlines()
# ///////////////////////////////////////////////////////// #

# the training steps
if RESTORE:
	latest_file = latest_file(path+'/')
	print("Restoring model from:", latest_file)
	AV_model = load_model(latest_file,custom_objects={"tf": tf})
	info = latest_file.strip().split('-')
	initial_epoch = int(info[-2])
else:
	AV_model = AV.AV_model(people_num)

train_generator = AVGenerator(trainfile,database_dir_path= database_dir_path, batch_size=batch_size, shuffle=True)
val_generator = AVGenerator(valfile,database_dir_path=database_dir_path, batch_size=batch_size, shuffle=True)

if NUM_GPU > 1:
	parallel_model = ModelMGPU(AV_model,NUM_GPU)
	adam = optimizers.Adam()
	loss = audio_loss(gamma=gamma_loss,beta=beta_loss,num_speaker=people_num)
	parallel_model.compile(loss=loss,optimizer=adam, metrics=['accuracy'])
	print(AV_model.summary())
	parallel_model.fit_generator(generator=train_generator,
						   validation_data=val_generator,
						   epochs=epochs,
						   workers = workers,
						   use_multiprocessing= use_multiprocessing,
						   callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
						   initial_epoch=initial_epoch
						   )
if NUM_GPU <= 1:
	adam = optimizers.Adam()
	loss = audio_loss(gamma=gamma_loss,beta=beta_loss, num_speaker=people_num)
	AV_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
	print(AV_model.summary())
	AV_model.fit_generator(generator=train_generator,
						   validation_data=val_generator,
						   epochs=epochs,
						   workers = workers,
						   use_multiprocessing= use_multiprocessing,
						   callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
						   initial_epoch=initial_epoch
						   )


























