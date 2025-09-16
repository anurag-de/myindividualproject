# AI driven energy-efficient approach for human detection using an ultrasonic sensor system

## Introduction:

The project focuses on a real time human detection system using SRF02 ultrasonic sensor and Red Pitaya STEM lab board inside an office environment and thereby activating an LED on red pitaya board if human presence is detected. The system is controlled by an user-friendly graphical user interface built with PyQt6. When turned on, the system scans for human presence twice every second. It has two modes, "Activity Detection" mode, where the sensor scans for sudden changes in the distance between the sensor and floor that can be caused by something/someone coming inside the monitoring zone of the sensor, and "CNN Classify" mode, where a pre-trained PyTorch CNN model is used to detect presence of human for a certain period of time (e.g. 15 seconds). The purpose of using two modes in the system is to promote energy efficiency. A user interface is also provided to the user for adjusting the settings. The picture below shows the FIUS sensor system with SRF02 and red pitaya board.

![Alt Text](https://github.com/anurag-de/myindividualproject/blob/main/assets/pics/red-pitaya-fius.jpg)

In "Activity Detection" mode, when something/someone moves into monitoring zone of the sensor, the LED is turned on as shown below. However, if it is an inanimate object that triggered this mode, the CNN model will detect that it is non-human, so the LED will be turned off. However, the "CNN Classify" mode stays on for 4 times the timeout period, and if still a human is not detected, the system shifts to "Activity Detection" mode. 

![Alt Text](https://github.com/anurag-de/myindividualproject/blob/main/assets/gifs/activity-trigger.gif)

Also, in "CNN Classify" mode, if human is detected the LED stays on as shown below, until it does not detect human for a certain timeout period.

![Alt Text](https://github.com/anurag-de/myindividualproject/blob/main/assets/gifs/cnn-human-detection.gif)

The picture below shows the LED7 on red-pitaya board that is on when triggered by human presence or sudden changes in monitored environment.

![Alt Text](https://github.com/anurag-de/myindividualproject/blob/main/assets/pics/led-on.jpg)

## Requirements:

For running the detection system we need the following python libraries:

```shell
pip install PyQt6 pyqtgraph numpy torch scipy paramiko
```
For training the model, we also need these additional python libraries:

```shell
pip install pandas tqdm matplotlib seaborn scikit-learn jupyter
```

## Dataset used:

The ADC data captured from ultrasonic sensor in csv format using data acquisition software contains 25017 data points, of which the first 5500 is ignored as it contains the sending signal. The dataset that has been used to train and validate the custom PyTorch CNN classification model is hosted on Hugging Face. Links to them are provided below -

- Link to my training data and validation data files hosted in Hugging Face: [Link](https://huggingface.co/datasets/anurag-de/redpitaya-data/tree/main/csvdatafiles)
- Link to the processed spectrogram files: [Link](https://huggingface.co/datasets/anurag-de/redpitaya-data/tree/main/processednpyfiles)

The dataset has been processed in such a way that it can capture most of the possible scenarios of an office environment. The training data contains data from three persons and validation data contain data from another person who is not in the training data. 

Training data have 60000 human and 60000 non-human data points (Total: 120000 data points), while validation data have 6000 human and 6000 non-human data points (Total: 12000 data points).

## Model training:

For model training, in the beginning, csv files from training data and validation data were preprocessed by ignoring the first 5500 data points and then converting them into spectrogram using STFT. Basically we are converting 1D signal into 2D spectrogram to capture signal's frequency over time. The resulting spectrograms and their corresponding labels were save as numpy files. 

- Link to preprocessing jupyter notebook for training data: [Link](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/preprocess.ipynb)
- Link to preprocessing jupyter notebook for validation data: [Link](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/preprocess-validation-data.ipynb)

The preprocessed numpy files are used to train the 2D CNN model using PyTorch in two ways. In the first way, hold-out validation was done by loading only the preprocessed training dataset and splitted into 80% training and 20% validation data. The model was trained for 15 epochs and a validation accuracy of **99.61%** was achieved. 

- Link to the hold-out validation training jupyter notebook: [Link](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/training(hold-out-validation).ipynb)

In another way, both the preprocessed training and validation datasets were loaded and trained for 50 epochs with early stopping mechanism. The best performing model got validation accuracy of **95.79%**. 

- Link to the training jupyter notebook: [Link](https://github.com/anurag-de/myindividualproject/blob/main/src/cnntraining/train(full).ipynb)

We chose the model trained in the second way, which has been tested with a seperate validation dataset. The 2D CNN model used for classification has the following structure: 

```python
SignalCNN2D(
  (conv_block1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv_block2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc_block): Sequential(
    (0): Linear(in_features=38912, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=128, out_features=2, bias=True)
  )
)

```

The model has total trainable parameters of 4,986,018. 

## GUI & usage instructions:

To run the monitoring system, run the following:

```python
python ui.py  
```

Please make sure that the trained model should be present in the folder "myfullmodel", and that folder should be in the same directory as "ui.py".

Once the system starts, the graphical user interface looks something like this:

![Alt text](https://github.com/anurag-de/myindividualproject/blob/main/assets/pics/initial-start-window.png)

To start the system, click on "Start Sensor" button. This establishes an SSH connection with the red pitaya device and starts the compiled dma_with_udp_faster.c code inside red pitaya board for data acquisition. After this, an UDP handshake is performed to confirm the connection. In the beginning, "Activity Detection" mode starts. Once the stream is stable, clicking on "Enable Detection" starts the detection process. The system analyzes 2 signals per second.

![Alt text](https://github.com/anurag-de/myindividualproject/blob/main/assets/pics/activity-detection-mode.png)

The settings section has the following options:

- Timeout (s): Time-period set by the user as to how long the LED and the activity status button: "HUMAN DETECTED" or "ACTIVITY DETECTED" should stay on if any sudden activity or human is detected from the last signal. (Default: 15s)
- Debounce: The number of consequtive signals that must exceed the confidence threshold for human detection (Default: 3). This is to prevent false positives.
- Confidence: The probability that CNN model predicts human presence (Default: 0.99). Anything over this probability will be considered as human.
- Color: Color of the signal displayed (Default: White).
- Movement Thresh (idx): Minimum shift in the signal's peak index required to trigger the "Activity Detection" mode and activate the CNN for more detailed classification. A lower value makes the system more sensitive to small movements.(Default: 2000)

The detailed status section displays the following:

- App: Displays current status and updates from the application.
- Sensor: Shows real-time status messages.
- Stream: Shows the health of the incoming data packet stream.
- Time: Shows the current system time.
- Human Prob: Shows the human probability (from 0.0 to 1.0) from the CNN.
- Debounce: Shows current debounce count against the required number of hits.
- Complete: Number of complete packets received since start of session.
- Broken: Number of incomplete/broken/out-of-order packets received since start of session.
- Movement Buffer: Number of peak locations stored from the last four signals for checking significant movement.
- Peak Index: Index of the highest peak found in last signal.

When the "Activity Detection" mode is running and "Enable Detection" button is pressed, the system receives full 25,000-sample signals and will analyze them in real-time for sudden changes in distance between the sensor and the reflecting surface (e.g: Floor). The peak and their index of all the incoming signals are analyzed. If the peak shifts significantly over a short time window (4 signals) over movement threshold index, it's flagged as "activity." and the LED7 on the red-pitaya board turns on. The system then enters "CNN Classify" mode.

![Alt text](https://github.com/anurag-de/myindividualproject/blob/main/assets/pics/activity-detected.png)

In the "CNN Classify" mode, the application uses the pre-trained 2D CNN PyTorch model. If it does not detect any human presence, for instance if the movement is caused by movement of an object like pushed chair, the LED will be turned off and also shown in the GUI as shown below. The application will be in this mode for exactly four times the timeout period set by the user, and by then, if any movement is not found, "Activity Detection" mode starts again.

![Alt text](https://github.com/anurag-de/myindividualproject/blob/main/assets/pics/cnn-classify-mode.png)

However in "CNN Classify" mode, if any human presence is detected, it is shown in the GUI as shown below. The LED also stays on as long as the human is present. After this, if human presence is not found, the above procedure happens again.

![Alt text](https://github.com/anurag-de/myindividualproject/blob/main/assets/pics/human-detected.png)

Once the "Start Sensor" is pressed, the application always plots the raw ADC signal in real-time using pyqtgraph for immediate visual feedback, until the "Stop Sensor" is pressed. 





