# AI driven energy-efficient approach for human detection using an ultrasonic sensor system

# Introduction:

The project focuses on using a red-pitaya ultrasonic sensor to detect presence of something or someone inside an office environment, whether animate or inanimate, entering it's monitoring zone, and thereby activating an LED on red pitaya board. When turned on, the system scans for human presence twice every second. It has two modes, "Activity Detection" mode, where the sensor scans for sudden changes in the distance between the sensor and floor, and "CNN Classify" mode, where a pre-trained PyTorch CNN model is used to detect presence of human for a certain period of time (e.g. 15 seconds). The purpose of using two modes are energy efficiency. A user interface is also provided to the user for adjusting the settings.  





