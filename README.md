# Google Audiovisual Model

## Description

Adapted version of the model for for Part IV Project #80 (2019). Commands are adpated to be run on windows.

A modified version of https://github.com/bill9800/speech_separation (adapted for Part IV Project #80).

## Dependencies

To run this project fully, the following libraries and tools are required.

### Language and version:

Python 3.6

### Download and install:

**Use pip install:**

* keras

* tensorflow

* librosa

* youtube-dl

* pytest

* sox

To install all the required libraries listed above, run the following command from the *ai_project* folder:

```
python -m pip install -r requirements.txt
```

**Install manually:**

* ffmpeg

* cmake

**Ensure that the following are in your system path environment variables:**

* ...\Python\Scripts

* ...\ffmpeg\bin

* ...\sox

## Instructions for Running

### Dataset

Follow the instructions on the [README](https://github.com/ktam069/Audio-visual_speech_separation_basic/tree/master/data) in the data folder.

Steps 2-7* for data downloading can be run by calling:
```
python download_dataset.py
```
from within the data folder.

### Trainning the Model

After having downloaded a range of data into the data folder, the audiovisual model can be trained or ran.

From within **/model/model_v2**, run the following to train the model:

```python
python AV_train.py
```

### Using the Trained Model

Put the saved H5 model file into the **/model/model_v2** directory, and change the model path parameter in **predict_video.py**	to the correct name.

From within **/model/model_v2**, run the following to download a test video:

```python
python download_test_video.py
```

From within **/model/model_v2**, run the following to run the demo:

```python
python predict_video.py
```

The outputted clean audio files will be found in the **pred** folder.
