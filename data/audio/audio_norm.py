import librosa
import os
import numpy as np
import scipy.io.wavfile as wavfile

# Range of data to normalise
start_id = 0
end_id = 2

def norm(s_id=start_id, e_id=end_id):
	RANGE = (s_id,e_id)

	if(not os.path.isdir('norm_audio_train')):
		os.mkdir('norm_audio_train')

	for num in range(RANGE[0],RANGE[1]):
		path = 'audio_train/trim_audio_train%s.wav'% num
		norm_path = 'norm_audio_train/trim_audio_train%s.wav'% num
		if (os.path.exists(path)):
			audio,_= librosa.load(path,sr=16000)
			max = np.max(np.abs(audio))
			norm_audio = np.divide(audio,max)
			wavfile.write(norm_path,16000,norm_audio)


if __name__ == '__main__':
	norm()















