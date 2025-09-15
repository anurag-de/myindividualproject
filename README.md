# AI driven energy-efficient approach for human detection using an ultrasonic sensor system

## Introduction:

The project focuses on using a red-pitaya ultrasonic sensor to detect presence of something or someone inside an office environment, whether animate or inanimate, entering it's monitoring zone, and thereby activating an LED on red pitaya board. When turned on, the system scans for human presence twice every second. It has two modes, "Activity Detection" mode, where the sensor scans for sudden changes in the distance between the sensor and floor that can be caused by something/someone coming inside the monitoring zone of th sensor, and "CNN Classify" mode, where a pre-trained PyTorch CNN model is used to detect presence of human for a certain period of time (e.g. 15 seconds). The purpose of using two modes in the system is to promote energy efficiency. A user interface is also provided to the user for adjusting the settings.

In "Activity Detection" mode, when something/someone moves into monitoring zone of the sensor, the LED is turned on as shown below. However, if it is an inanimate object that triggered this mode, the CNN model will detect that it is non-human, so the LED will be turned off. However, the "CNN Classify" mode stays on for 4 times the timeout period, and if still a human is not detected, the system shifts to "Activity Detection" mode. 

![Alt Text](https://github.com/anurag-de/myindividualproject/blob/main/assets/gifs/activity-trigger.gif)

However, if human is detected the LED stays on as shown below until it does not detect human for a certain timeout period as shown below.

![Alt Text](https://github.com/anurag-de/myindividualproject/blob/main/assets/gifs/cnn-human-detection.gif)

## Requirements:

For running the detection system we need the following python libraries:

```shell
pip install PyQt6 pyqtgraph numpy torch scipy paramiko
```
For running the model training we also need the additional python libraries:

```shell
pip install pandas tqdm matplotlib seaborn scikit-learn jupyter
```

## Dataset used:

The ADC data captured from ultrasonic sensor in csv format using data acquisition software contains 25017 data points, of which the first 5500 is ignored as it contains the sending signal. The dataset that has been used to train and validate the custom PyTorch CNN classification model is hosted on Hugging Face. Links to them are provided below -

Link to my training data and validation data files hosted in Hugging Face: [Link](https://huggingface.co/datasets/anurag-de/redpitaya-data/tree/main/csvdatafiles)
Link to the processed spectrogram files: [Link](https://huggingface.co/datasets/anurag-de/redpitaya-data/tree/main/processednpyfiles)

The dataset has been processed in such a way that it can capture most of the possible scenarios of an office environment. The training data contains data from three persons and validation data contain data from another person who is not in the training data. 

## Model training:

For model training, in the beginning, csv files from training data and validation data were preprocessed by ignoring the first 5500 data points and then converting them into spectrogram using STFT. Basically we are converting 1D signal into 2D spectrogram to capture signal's frequency over time. The resulting spectrograms and their corresponding labels were save as numpy files. 

Links to preprocessing jupyter notebooks: ![Code1](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/preprocess.ipynb) & ![Code2](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/preprocess-validation-data.ipynb)

The preprocessed numpy files are used to train the 2D CNN model using PyTorch in two ways. In the first way, hold-out validation was done by loading only the preprocessed training dataset and splitted into 80% training and 20% validation data. The model was trained for 15 epochs and a validation accuracy of 99.61% was achieved. 

Link to the hold-out validation training jupyter notebook: ![Code](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/training(hold-out-validation).ipynb)

In another way, both the preprocessed training and validation datasets were loaded and trained for 50 epochs with early stopping mechanism. The best performing model got validation accuracy of 95.79%. 

Link to the training jupyter notebook: ![Code](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/train(full).ipynb)






