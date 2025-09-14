# ==============================================================================
#
# Human Presence Detector UI using Ultrasonic Red Pitaya Sensor & PyTorch CNN v1.0
#
# Author: Anurag
#
# ==============================================================================

# --- Core Python and System Libraries ---
import sys
import time
import struct
import warnings

# --- GUI and Plotting Libraries (PyQt6 and pyqtgraph) ---
import pyqtgraph as pg
from PyQt6.QtCore import QRunnable, pyqtSlot, QThreadPool, QObject, pyqtSignal, QTimer, Qt
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QMainWindow, QGridLayout,
                             QHBoxLayout, QVBoxLayout, QLabel, QGroupBox, QSpinBox,
                             QCheckBox, QDoubleSpinBox, QComboBox, QFrame)

# --- Machine Learning and Signal Processing Libraries ---
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

# --- Network Communication Libraries ---
import socket
import paramiko

# --- Suppress common warnings for a cleaner console output ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration Constants ---
# These constants define the parameters for signal processing and the ML model.
FIXED_LENGTH = 19517  # The exact signal length the CNN model expects.
N_FFT = 256           # The number of FFT points for the STFT spectrogram.
HOP_LENGTH = 128      # The hop length for the STFT.

# --- Dark Mode Stylesheet for a modern UI ---
# A single string containing all the CSS-like styling for the PyQt6 application.
DARK_STYLE = """
QWidget { background-color: #2e2e2e; color: #ffffff; font-family: Arial; }
QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 1ex; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }
QPushButton { background-color: #555; border: 1px solid #666; padding: 6px; border-radius: 3px; }
QPushButton:hover { background-color: #666; }
QPushButton:pressed { background-color: #444; }
QPushButton:disabled { background-color: #404040; color: #888; }
QLabel { background-color: transparent; }
QComboBox { background-color: #444; border: 1px solid #555; padding: 5px; border-radius: 3px; }
QCheckBox { color: #ffffff; spacing: 5px; }
QCheckBox::indicator { border: 1px solid #555; width: 15px; height: 15px; background-color: #444; }
QCheckBox::indicator:hover { border-color: #666; }
QCheckBox::indicator:checked {
    image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik05IDE2LjE3TDQuODMgMTIgMy40MSAxMy40MSA5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz48L3N2Zz4=);
    background-color: #2ecc71;
    border-color: #2ecc71;
}
QFrame[frameShape="4"] { border: 1px solid #444; }
"""

# ==============================================================================
# PYTORCH MODEL DEFINITION & PREPROCESSING
# ==============================================================================
class SignalCNN2D(nn.Module):
    """
    A 2D Convolutional Neural Network for classifying signal spectrograms.
    The architecture consists of two convolutional blocks followed by a fully connected classifier.
    """
    def __init__(self, input_shape):
        super(SignalCNN2D, self).__init__()
        c, h, w = input_shape
        self.conv_block1 = nn.Sequential(nn.Conv2d(c, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        flattened_size = self._get_conv_output_size(input_shape)
        self.fc_block = nn.Sequential(nn.Linear(flattened_size, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2))
    
    def _get_conv_output_size(self, shape):
        """Helper function to calculate the output size of the conv layers for the linear layer."""
        with torch.no_grad(): 
            x = torch.zeros(1, *shape)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            return x.numel()
            
    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

def preprocess_signal_for_pytorch(signal_chunk):
    """
    Transforms a raw 1D signal chunk into a 2D spectrogram tensor suitable for the CNN.
    Steps:
    1. Pad or truncate the signal to FIXED_LENGTH.
    2. Compute the Short-Time Fourier Transform (STFT) to get a complex spectrogram.
    3. Take the absolute value to get the magnitude spectrogram.
    4. Add batch and channel dimensions to match the model's expected input shape (N, C, H, W).
    """
    try:
        # Ensure the signal is the correct length
        # Pad with zeros if too short or truncate if too long, to avoid errors
        if len(signal_chunk) > FIXED_LENGTH: signal_chunk = signal_chunk[:FIXED_LENGTH]
        elif len(signal_chunk) < FIXED_LENGTH:
            padding = np.zeros(FIXED_LENGTH - len(signal_chunk), dtype=np.float32)
            signal_chunk = np.concatenate((signal_chunk, padding))
        
        # Create spectrogram using STFT
        spec = torch.stft(torch.from_numpy(signal_chunk).float(), n_fft=N_FFT, hop_length=HOP_LENGTH, window=torch.hann_window(N_FFT), return_complex=True)
        
        # Return the magnitude spectrogram with batch and channel dimensions
        return torch.abs(spec).unsqueeze(0).unsqueeze(0)
    except Exception as e: 
        print(f"Preprocessing failed: {e}")
        return None

# ==============================================================================
# SENSOR COMMUNICATION CLASS
# ==============================================================================
class RedPitayaSensor:
    """Handles all low-level communication with the Red Pitaya device."""
    def __init__(self):
        # --- Network Configuration ---
        self.hostIP = "169.254.148.148"
        self.data_port = 61231
        self.ssh_port = 22
        self.server_address_port = (self.hostIP, self.data_port)

        # --- Data Protocol Configuration ---
        self.size_of_raw_adc = 25000
        self.buffer_size = (25000 + 17) * 4 # Max expected UDP packet size

        # --- State and Clients ---
        self.sensor_status_message = "Waiting to Connect..."
        self.udp_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.udp_client_socket.settimeout(2.0) # Timeout to prevent indefinite blocking
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) # Auto-accept SSH key
        
    def connect_ssh(self):
        """Establishes an SSH connection to the device if not already active."""
        if not self.is_ssh_connected():
            try: self.ssh_client.connect(self.hostIP, self.ssh_port, "root", "root", timeout=5)
            except Exception as e: raise ConnectionError(f"SSH connection failed: {e}")
            
    def is_ssh_connected(self):
        """Checks if the SSH transport is active."""
        return self.ssh_client.get_transport() and self.ssh_client.get_transport().is_active()
        
    def disconnect_ssh(self):
        """Closes the SSH connection if it's open."""
        if self.is_ssh_connected(): self.ssh_client.close()
            
    def give_ssh_command(self, command):
        """Executes a command on the Red Pitaya via SSH, auto-connecting if needed."""
        try:
            if not self.is_ssh_connected(): self.connect_ssh()
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            if (error := stderr.read().decode().strip()) and stdout.channel.recv_exit_status() != 0: 
                print(f"SSH Error: {error}")
            return stdout.read().decode()
        except Exception as e: raise ConnectionError(f"SSH command failed ('{command}'): {e}")
            
    def set_sensor_message(self, message):
        """Sets the status message to be displayed in the GUI."""
        self.sensor_status_message = message
        
    def get_sensor_status_message(self):
        """Retrieves the current status message."""
        return self.sensor_status_message
        
    def send_msg_to_server(self, msg):
        """Sends a UDP message to the sensor."""
        self.udp_client_socket.sendto(str.encode(msg), self.server_address_port)
        
    def get_data_info_from_server(self):
        """Performs the initial handshake to get data metadata from the sensor."""
        self.send_msg_to_server("-i 1") # Command to request info
        try:
            packet = self.udp_client_socket.recv(self.buffer_size)
            # Unpack binary data from the response packet
            self.header_length = int(struct.unpack('@f', packet[:4])[0])
            self.total_data_blocks = int(struct.unpack('@f', packet[56:60])[0])
            if self.total_data_blocks > 0: 
                self.set_sensor_message(f"Connected to {self.hostIP}:{self.data_port}")
                return True
            else: raise ConnectionError("Handshake failed: 0 data blocks.")
        except socket.timeout: raise ConnectionError("Handshake timed out.")
            
    def get_data_from_server(self):
        """Retrieves a full signal chunk from the sensor over UDP."""
        if self.total_data_blocks == 0: raise ConnectionError("Not connected.")
        ultrasonic_data = []
        self.set_sensor_message("Receiving...")
        # Loop to receive all data blocks that constitute one full signal
        for i in range(self.total_data_blocks):
            self.send_msg_to_server("-a 1") # Command to request a data block
            packet = self.udp_client_socket.recv(self.buffer_size)
            
            # Sanity check for packet order
            if i != int(struct.unpack('@f', packet[60:64])[0]): raise ValueError("Data sync error.")
            
            # Unpack the ADC values (signed short integers) from the packet payload
            for d in struct.iter_unpack('@h', packet[self.header_length:]): 
                ultrasonic_data.append(d[0])

        if len(ultrasonic_data) != self.size_of_raw_adc: 
            raise ValueError("Data length mismatch. Detection will not work.")
        
        return np.array(ultrasonic_data, dtype=np.float32)

# ==============================================================================
# WORKER THREADS
# ==============================================================================
class GenericRunnable(QRunnable):
    """A generic, reusable QRunnable that can run any target function with args."""
    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self.target, self.args, self.kwargs = target, args, kwargs
    @pyqtSlot()
    def run(self):
        self.target(*self.args, **self.kwargs)

class StartupWorker(QRunnable):
    """
    Handles the initial sensor connection and setup sequence in a background thread.
    Emits signals to notify the main window of progress, success, or failure.
    """
    class Signals(QObject):
        status_updated = pyqtSignal(str, bool)
        startup_successful = pyqtSignal()
        startup_failed = pyqtSignal(str)

    def __init__(self, rp_sensor):
        super().__init__()
        self.rp_sensor, self.signals = rp_sensor, self.Signals()

    @pyqtSlot()
    def run(self):
        """Executes the startup sequence."""
        try:
            # Step 1: Establish SSH and reset LED state
            # Specifically reset the LED7 to OFF state
            self.signals.status_updated.emit("Establishing SSH...", False); self.rp_sensor.connect_ssh()
            self.signals.status_updated.emit("Resetting LED...", False); self.rp_sensor.give_ssh_command("/opt/redpitaya/bin/monitor 0x40000030 0x0")
            
            # Step 2: Start the data acquisition program on the Red Pitaya
            self.signals.status_updated.emit("Sending start command...", False); self.rp_sensor.give_ssh_command("cd /usr/RedPitaya/Examples/C && ./dma_with_udp_faster &"); time.sleep(1)
            
            # Step 3: Perform UDP handshake
            self.signals.status_updated.emit("Establishing handshake...", False); self.rp_sensor.get_data_info_from_server()
            self.signals.status_updated.emit("Sensor connected. Warming up...", False); time.sleep(2)
            
            # Step 4: Signal success
            self.signals.startup_successful.emit()
        except Exception as e:
            self.signals.startup_failed.emit(str(e))

class Worker(QRunnable):
    """
    The main data processing worker. Runs in a continuous loop to fetch data,
    perform detection, and emit results to the GUI.
    """
    class Signals(QObject):
        raw_plot_ready = pyqtSignal(np.ndarray)
        prediction_made = pyqtSignal(float)
        activity_detected = pyqtSignal(float)
        mode_changed = pyqtSignal(str)
        status_updated = pyqtSignal(str, bool)
        finished = pyqtSignal()
        increment_total = pyqtSignal()
        increment_broken = pyqtSignal()

    def __init__(self, rp_sensor, threadpool, model):
        super().__init__()
        self.rp_sensor, self.threadpool, self.model = rp_sensor, threadpool, model
        self.signals = self.Signals()
        
        # --- Worker State ---
        self.is_running = True          # Flag to control the main loop
        self.detection_enabled = False  # Flag to enable/disable detection logic
        self.detection_mode = 'peak'    # Current mode: 'peak' (activity) or 'cnn' (classify)
        self.cnn_only_mode = False      # If true, bypasses peak detection
        
        # --- Peak Detection Parameters ---
        # These can be adjusted for sensitivity
        self.min_peak_height = 100.0
        self.min_peak_prominence = 50.0
        self.movement_index_threshold = 2000
        
        # --- Movement Detection Buffer ---
        self.movement_buffer = [] 
        self.BUFFER_SIZE = 4      
        self.last_peak_index = -1

    def set_detection_mode(self, mode):
        """Switches the detection mode between 'peak' and 'cnn'."""
        if mode != self.detection_mode:
            self.detection_mode = mode
            self.signals.mode_changed.emit(mode)
            if mode == 'peak':
                self.reset_movement_buffer()

    def set_cnn_only_mode(self, enabled):
        """Enables or disables the CNN-only mode."""
        self.cnn_only_mode = enabled
        if enabled:
            self.set_detection_mode('cnn')
            self.signals.status_updated.emit("CNN-only mode active. Activity detection bypassed.", False)
        else:
            self.set_detection_mode('peak')
            self.signals.status_updated.emit("Two-stage detection active (Activity + CNN).", False)

    def reset_movement_buffer(self):
        """Clears the buffer used for peak-based movement detection."""
        self.movement_buffer.clear() 
        self.signals.status_updated.emit("Activity movement buffer reset.", False)

    @pyqtSlot()
    def run(self):
        """The main processing loop of the worker thread."""
        target_interval = 0.5  # Looping every 500ms
        while self.is_running:
            loop_start_time = time.time()
            try:
                # 1. Get raw data from the sensor
                raw_numpy_data = self.rp_sensor.get_data_from_server()
                self.signals.increment_total.emit()
                self.signals.raw_plot_ready.emit(raw_numpy_data)
                
                # 2. Create two separate signals for analysis
                # Instead of 5500, 5483 is used as model was trained with that offset
                signal_for_peak_detection = raw_numpy_data.flatten()
                signal_for_cnn = raw_numpy_data.flatten()[5483:5483 + FIXED_LENGTH]

                # 3. Run detection logic if enabled
                if self.detection_enabled and len(signal_for_peak_detection) > 0:
                    if self.cnn_only_mode:
                        self.run_cnn_classify(signal_for_cnn)
                    elif self.detection_mode == 'peak':
                        # Pass the full signal to peak detection
                        self.run_peak_detection(signal_for_peak_detection)
                    elif self.detection_mode == 'cnn':
                        # Pass the correctly sized signal to the CNN
                        self.run_cnn_classify(signal_for_cnn)

            except Exception as e:
                # Handle errors (e.g., connection lost) and report them
                self.signals.increment_broken.emit()
                self.signals.status_updated.emit(str(e), True)
            finally:
                # Sleep to maintain the target interval
                if (sleep_time := target_interval - (time.time() - loop_start_time)) > 0: 
                    time.sleep(sleep_time)
        self.signals.finished.emit()

    def run_peak_detection(self, signal_data):
        """
        First-stage detection: Detects general activity by analyzing shifts in the signal's
        most prominent peak over a short time window (4 signals / 2 seconds).
        """
        peaks, _ = find_peaks(signal_data, height=self.min_peak_height, prominence=self.min_peak_prominence)
        
        absolute_peak_index = -1
        if len(peaks) > 0:
            # Find the index of the highest peak
            relative_peak_index = peaks[np.argmax(signal_data[peaks])]
            absolute_peak_index = relative_peak_index # No offset needed for full signal
            
        # Add the peak index to a short-term buffer
        self.last_peak_index = absolute_peak_index
        self.movement_buffer.append(absolute_peak_index)
        
        # Once the buffer is full, check for significant movement
        if len(self.movement_buffer) == self.BUFFER_SIZE:
            first_index = self.movement_buffer[0]
            fourth_index = self.movement_buffer[3]
            
            if first_index != -1 and fourth_index != -1:
                index_shift = abs(fourth_index - first_index)
                
                # If peak has shifted more than the threshold, it's considered "activity"
                if index_shift > self.movement_index_threshold:
                    self.signals.activity_detected.emit(1.0) # Emit activity signal
                    if not self.cnn_only_mode:
                        self.set_detection_mode('cnn') # Switch to CNN mode for classification
            
            self.movement_buffer.clear() # Reset buffer for next check

    def run_cnn_classify(self, signal_data):
        """
        Second-stage detection: Preprocesses the signal and runs it through the CNN
        to get a 'human' vs 'non-human' classification probability.
        """
        if len(signal_data) >= FIXED_LENGTH:
            spectrogram_tensor = preprocess_signal_for_pytorch(signal_data)
            if spectrogram_tensor is not None:
                self.predict(spectrogram_tensor)
        else:
            self.signals.status_updated.emit(f"Signal too short for CNN: {len(signal_data)}/{FIXED_LENGTH}", True)
            if not self.cnn_only_mode:
                self.set_detection_mode('peak') # Revert to peak mode if signal is invalid

    def predict(self, spectrogram_tensor):
        """Performs inference using the loaded PyTorch model."""
        if self.model is None: return
        try:
            with torch.no_grad(): # Disable gradient calculation for faster inference
                output = self.model(spectrogram_tensor.to(next(self.model.parameters()).device))
                # Emit the probability of the 'human' class (index 1)
                self.signals.prediction_made.emit(torch.softmax(output, dim=1)[0][1].item())
        except Exception as e: self.signals.status_updated.emit(f"Prediction error: {e}", True)

    def stop(self):
        """Stops the worker's main loop."""
        self.is_running = False
        
    def set_detection_enabled(self, enabled):
        """Enables or disables the detection logic via the GUI checkbox."""
        self.detection_enabled = enabled
        if enabled:
            self.reset_movement_buffer()

# ==============================================================================
# MAIN APPLICATION WINDOW
# ==============================================================================
class MainWindow(QMainWindow):
    """
    The main application window, responsible for all UI elements,
    state management, and orchestrating the worker threads.
    """
    STABLE_STREAM_THRESHOLD = 5 # Number of consecutive good signals to declare stream stable
    
    def __init__(self):
        super().__init__()
        # --- Core Components ---
        self.rp_sensor = RedPitayaSensor()
        self.threadpool = QThreadPool()
        self.worker, self.model = None, None
        
        # --- Application State ---
        self.last_human_detection_time = 0
        self.last_activity_time = 0
        self.last_cnn_mode_time = 0
        self.total_signals_count = 0
        self.broken_signals_count = 0
        self.debounce_counter = 0
        self.consecutive_good_signals = 0
        self.stream_is_stable = False
        self.led_state = "OFF" # Tracks the LED state to avoid redundant SSH commands
        
        # --- User-configurable Settings ---
        self.detection_timeout_s = 15
        self.detection_threshold = 0.99
        self.debounce_hits_required = 3
        self.color_map = {'White': 'w', 'Red': 'r', 'Green': 'g', 'Blue': 'b', 'Yellow': 'y', 'Cyan': 'c', 'Magenta': 'm'}
        
        # --- Initialization ---
        self.setWindowTitle("Human Presence Detector (using PyTorch CNN)"); self.resize(1200, 650)
        self.setup_ui()
        self.setup_model()
        self.setup_timers()

    def setup_ui(self):
        """Creates and arranges all the UI widgets."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Plot Widget ---
        self.plot_widget = pg.PlotWidget(title="Raw Signal (ADC)")
        self.plot_widget.showGrid(x=True, y=True)
        self.raw_signal_plot_item = self.plot_widget.plot(pen='w')
        main_layout.addWidget(self.plot_widget)

        bottom_deck_layout = QHBoxLayout()

        # --- Primary Status Group ---
        primary_status_group = QGroupBox("Primary Status")
        primary_status_layout = QVBoxLayout()
        self.human_detection_status_btn = QPushButton("NO HUMAN DETECTED")
        self.human_detection_status_btn.setMinimumHeight(50)
        self.human_detection_status_btn.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 18px;")
        self.activity_indicator_label = QLabel("ACTIVITY MODE - IDLE")
        self.activity_indicator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activity_indicator_label.setMinimumHeight(40)
        self.activity_indicator_label.setStyleSheet("background-color: #555; border-radius: 4px; font-weight: bold; font-size: 16px;")
        self.mode_label = QLabel("Mode: Activity Detection")
        self.mode_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 14px;")
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.led_indicator = QLabel("LED: OFF")
        self.led_indicator.setStyleSheet("background-color: #444; color: white; border-radius: 4px; font-weight: bold; font-size: 14px;")
        self.led_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.led_indicator.setMinimumHeight(30)
        primary_status_layout.addWidget(self.human_detection_status_btn)
        primary_status_layout.addWidget(self.activity_indicator_label)
        primary_status_layout.addWidget(self.mode_label)
        primary_status_layout.addWidget(self.led_indicator)
        primary_status_group.setLayout(primary_status_layout)
        bottom_deck_layout.addWidget(primary_status_group, 2)

        # --- Controls & Settings Group ---
        controls_and_settings_group = QGroupBox("Settings")
        controls_and_settings_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()
        self.start_sensor_btn = QPushButton("Start Sensor"); self.start_sensor_btn.clicked.connect(self.start_sensor_btn_handler)
        self.detection_chkbox = QCheckBox("Enable Detection"); self.detection_chkbox.toggled.connect(self.detection_checkbox_handler); self.detection_chkbox.setEnabled(False)
        self.stop_sensor_btn = QPushButton("Stop Sensor"); self.stop_sensor_btn.clicked.connect(self.stop_sensor_btn_handler); self.stop_sensor_btn.setEnabled(False)
        controls_layout.addWidget(self.start_sensor_btn); controls_layout.addWidget(self.detection_chkbox); controls_layout.addWidget(self.stop_sensor_btn)
        controls_and_settings_layout.addLayout(controls_layout)
        self.detection_mode_btn = QPushButton("Switch to CNN-only Mode")
        self.detection_mode_btn.clicked.connect(self.toggle_detection_mode)
        self.detection_mode_btn.setEnabled(False)
        controls_and_settings_layout.addWidget(self.detection_mode_btn)
        controls_and_settings_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("Timeout (s):"), 0, 0); self.timeout_spinbox = QSpinBox(); self.timeout_spinbox.setRange(1, 120); self.timeout_spinbox.setValue(self.detection_timeout_s); self.timeout_spinbox.valueChanged.connect(lambda v: setattr(self, 'detection_timeout_s', v)); settings_layout.addWidget(self.timeout_spinbox, 0, 1)
        settings_layout.addWidget(QLabel("Confidence:"), 1, 0); self.prob_spinbox = QDoubleSpinBox(); self.prob_spinbox.setRange(0.5, 1.0); self.prob_spinbox.setSingleStep(0.01); self.prob_spinbox.setValue(self.detection_threshold); self.prob_spinbox.valueChanged.connect(lambda v: setattr(self, 'detection_threshold', v)); settings_layout.addWidget(self.prob_spinbox, 1, 1)
        settings_layout.addWidget(QLabel("Debounce:"), 0, 2); self.debounce_spinbox = QSpinBox(); self.debounce_spinbox.setRange(1, 20); self.debounce_spinbox.setValue(self.debounce_hits_required); self.debounce_spinbox.valueChanged.connect(self.update_debounce_setting); settings_layout.addWidget(self.debounce_spinbox, 0, 3)
        settings_layout.addWidget(QLabel("Color:"), 1, 2); self.color_combobox = QComboBox(); self.color_combobox.addItems(self.color_map.keys()); self.color_combobox.currentTextChanged.connect(self.update_signal_color); settings_layout.addWidget(self.color_combobox, 1, 3)
        settings_layout.addWidget(QLabel("Movement Thresh (idx):"), 2, 0)
        self.movement_threshold_spinbox = QSpinBox()
        self.movement_threshold_spinbox.setRange(50, 5000)
        self.movement_threshold_spinbox.setSingleStep(50)
        self.movement_threshold_spinbox.setValue(2000)
        self.movement_threshold_spinbox.valueChanged.connect(self.update_movement_threshold)
        settings_layout.addWidget(self.movement_threshold_spinbox, 2, 1)
        controls_and_settings_layout.addLayout(settings_layout)
        controls_and_settings_group.setLayout(controls_and_settings_layout)
        bottom_deck_layout.addWidget(controls_and_settings_group, 3)

        # --- Detailed Status Group ---
        status_groupbox = QGroupBox("Detailed Status")
        status_layout = QGridLayout()
        self.app_status_label = QLabel("App: Initializing, please wait..."); self.sensor_status_label = QLabel("Sensor: Waiting")
        self.stream_status_label = QLabel("Stream: Idle"); self.datetime_label = QLabel("Time: N/A")
        self.probability_label = QLabel("Human Prob: N/A"); self.debounce_label = QLabel(f"Debounce: 0/{self.debounce_hits_required}")
        self.total_signals_label = QLabel("Complete: 0"); self.broken_signals_label = QLabel("Broken: 0")
        self.movement_info_label = QLabel("Movement Buffer: 0/4")
        self.peak_index_label = QLabel("Peak Index: N/A")
        status_layout.addWidget(self.app_status_label, 0, 0, 1, 2)
        status_layout.addWidget(self.sensor_status_label, 1, 0, 1, 2)
        status_layout.addWidget(self.stream_status_label, 2, 0); status_layout.addWidget(self.datetime_label, 2, 1)
        status_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine), 3, 0, 1, 2)
        status_layout.addWidget(self.probability_label, 4, 0); status_layout.addWidget(self.debounce_label, 4, 1)
        status_layout.addWidget(self.total_signals_label, 5, 0); status_layout.addWidget(self.broken_signals_label, 5, 1)
        status_layout.addWidget(self.movement_info_label, 6, 0)
        status_layout.addWidget(self.peak_index_label, 6, 1)
        status_groupbox.setLayout(status_layout)
        bottom_deck_layout.addWidget(status_groupbox, 2)

        main_layout.addLayout(bottom_deck_layout)
        main_layout.setStretch(0, 5)
        main_layout.setStretch(1, 5)

    def update_movement_threshold(self, value):
        """Updates the movement threshold in the worker when the spinbox value changes."""
        if self.worker: self.worker.movement_index_threshold = value

    def toggle_detection_mode(self):
        """Switches between 'Activity+CNN' and 'CNN-only' modes."""
        if not self.worker: return
        cnn_only_mode = not self.worker.cnn_only_mode
        self.worker.set_cnn_only_mode(cnn_only_mode)
        if cnn_only_mode:
            self.detection_mode_btn.setText("Switch to Activity+CNN Mode")
            self.update_app_status("CNN-only mode activated", False)
        else:
            self.detection_mode_btn.setText("Switch to CNN-only Mode")
            self.update_app_status("Activity+CNN mode activated", False)

    def setup_model(self):
        """Loads the pre-trained PyTorch model from disk."""
        try:
            # Calculate the expected spectrogram dimensions
            spec_h = (N_FFT // 2) + 1
            spec_w = (FIXED_LENGTH // HOP_LENGTH) + 1
            
            # Instantiate and load the model
            self.model = SignalCNN2D((1, spec_h, spec_w))
            model_path = 'myfullmodel/best_model.pth'
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval() # Set model to evaluation mode
            self.update_app_status(f"PyTorch model loaded from '{model_path}'.", False)
        except Exception as e: 
            self.update_app_status(f"Error loading model: {e}", True)
            self.model = None

    def setup_timers(self):
        """Initializes QTimers for periodic UI updates."""
        # Timer to check for detection timeouts
        self.detection_status_timer = QTimer()
        self.detection_status_timer.setInterval(1000) # 1 second
        self.detection_status_timer.timeout.connect(self.update_human_detection_status)
        self.detection_status_timer.start()

        # Timer to update sensor status message
        self.status_message_timer = QTimer()
        self.status_message_timer.setInterval(250) # 250 ms
        self.status_message_timer.timeout.connect(lambda: self.sensor_status_label.setText(f"Sensor: {self.rp_sensor.get_sensor_status_message()}"))
        self.status_message_timer.start()

        # Timer to update the clock
        self.datetime_timer = QTimer()
        self.datetime_timer.setInterval(1000) # 1 second
        self.datetime_timer.timeout.connect(self.update_datetime)
        self.datetime_timer.start()

    def start_sensor_btn_handler(self):
        """Handles the 'Start Sensor' button click."""
        # 1. Update UI state to 'starting'
        self.start_sensor_btn.setEnabled(False)
        self.stop_sensor_btn.setEnabled(True)
        
        # 2. Reset all state variables and counters
        self.total_signals_count, self.broken_signals_count, self.consecutive_good_signals = 0, 0, 0
        self.stream_is_stable = False
        self.update_app_status("Waiting for stable stream...", False)
        self.stream_status_label.setText("Stream: Unstable"); self.stream_status_label.setStyleSheet("color: #ff4757;")
        self.total_signals_label.setText("Complete: 0"); self.broken_signals_label.setText("Broken: 0")
        self.raw_signal_plot_item.clear()
        self.last_human_detection_time, self.last_activity_time, self.debounce_counter, self.last_cnn_mode_time = 0, 0, 0, 0
        self.update_human_detection_status()
        self.probability_label.setText("Human Prob: N/A")
        self.debounce_label.setText(f"Debounce: 0/{self.debounce_hits_required}")
        self.movement_info_label.setText("Movement Buffer: 0/4")
        self.peak_index_label.setText("Peak Index: N/A")
        self.led_state = "OFF"
        self.update_led_indicator()
        
        # 3. Launch the StartupWorker in the threadpool
        startup_worker = StartupWorker(self.rp_sensor)
        startup_worker.signals.status_updated.connect(self.update_app_status_conditionally)
        startup_worker.signals.startup_successful.connect(self.on_startup_successful)
        startup_worker.signals.startup_failed.connect(self.on_startup_failed)
        self.threadpool.start(startup_worker)

    @pyqtSlot()
    def on_startup_successful(self):
        """Slot executed when the StartupWorker succeeds. Launches the main Worker."""
        self.update_app_status_conditionally("Starting data acquisition...", False)
        if self.model:
            # Create and configure the main worker
            self.worker = Worker(self.rp_sensor, self.threadpool, self.model)
            self.worker.movement_index_threshold = self.movement_threshold_spinbox.value()
            
            # Connect all signals from the worker to the appropriate slots in the GUI
            self.worker.signals.raw_plot_ready.connect(self.update_raw_plot)
            self.worker.signals.prediction_made.connect(self.handle_prediction)
            self.worker.signals.status_updated.connect(self.update_app_status)
            self.worker.signals.increment_total.connect(self.increment_total_signals)
            self.worker.signals.increment_broken.connect(self.increment_broken_signals)
            self.worker.signals.activity_detected.connect(self.handle_activity_detection)
            self.worker.signals.mode_changed.connect(self.handle_mode_change)
            self.worker.signals.finished.connect(lambda: self.update_app_status("Acquisition stopped.", False))
            
            # Start the worker and enable relevant UI controls
            self.threadpool.start(self.worker)
            self.detection_chkbox.setEnabled(True)
            self.detection_mode_btn.setEnabled(True)
        else: 
            self.update_app_status("Cannot start: PyTorch model not loaded.", True)

    @pyqtSlot(str)
    def on_startup_failed(self, error_message):
        """Slot executed when the StartupWorker fails."""
        self.update_app_status(error_message, True)
        self.start_sensor_btn.setEnabled(True)
        self.stop_sensor_btn.setEnabled(False)

    def _stop_sensor_sequence(self):
        """Commands to gracefully stop the remote process and turn off the LED."""
        try:
            # Find and kill the data acquisition process on the Red Pitaya
            if pid := self.rp_sensor.give_ssh_command("pidof dma_with_udp_faster").strip(): 
                self.rp_sensor.give_ssh_command(f"kill {pid}")
            # Turn off the LED
            self.rp_sensor.give_ssh_command("/opt/redpitaya/bin/monitor 0x40000030 0x0")
        except Exception as e: print(f"Error during stop sequence: {e}")
        finally: 
            self.rp_sensor.disconnect_ssh() # Always ensure SSH is disconnected

    def stop_sensor_btn_handler(self):
        """Handles the 'Stop Sensor' button click."""
        self.detection_chkbox.setChecked(False)
        self.detection_chkbox.setEnabled(False)
        self.detection_mode_btn.setEnabled(False)
        if self.worker: self.worker.stop() # Tell the worker loop to exit
        
        # Run the stop sequence in a separate thread to avoid freezing the GUI
        self.threadpool.start(GenericRunnable(self._stop_sensor_sequence))
        
        # Update UI state
        self.update_app_status("Stop command issued.", False)
        self.rp_sensor.set_sensor_message("Sensor stopped.")
        self.stream_status_label.setText("Stream: Idle"); self.stream_status_label.setStyleSheet("color: white;")
        self.start_sensor_btn.setEnabled(True)
        self.stop_sensor_btn.setEnabled(False)
        self.led_state = "OFF"
        self.update_led_indicator()

    @pyqtSlot(bool)
    def detection_checkbox_handler(self, state):
        """Handles toggling the 'Enable Detection' checkbox."""
        if self.worker:
            self.worker.set_detection_enabled(state)
            self.update_app_status_conditionally(f"Detection {'enabled' if state else 'disabled'}.", False)
            if state: self.worker.reset_movement_buffer()

    @pyqtSlot()
    def increment_total_signals(self):
        """Increments the count of successfully received signals."""
        self.total_signals_count += 1
        self.total_signals_label.setText(f"Complete: {self.total_signals_count}")
        
        # Logic to determine if the data stream has become stable
        if not self.stream_is_stable:
            self.consecutive_good_signals += 1
            if self.consecutive_good_signals >= self.STABLE_STREAM_THRESHOLD:
                self.stream_is_stable = True
                self.update_app_status("Sensor stream is stable.", False)
                self.stream_status_label.setText("Stream: Stable")
                self.stream_status_label.setStyleSheet("color: #2ecc71;")

    @pyqtSlot()
    def increment_broken_signals(self):
        """Increments the count of broken/missed signals and resets stability."""
        self.broken_signals_count += 1
        self.broken_signals_label.setText(f"Broken: {self.broken_signals_count}")
        self.consecutive_good_signals = 0 # Reset consecutive counter
        if self.stream_is_stable: 
            self.stream_is_stable = False
            self.update_app_status("Stream unstable!", True)
            self.stream_status_label.setText("Stream: Unstable")
            self.stream_status_label.setStyleSheet("color: #ff4757;")

    @pyqtSlot(np.ndarray)
    def update_raw_plot(self, y_data):
        """Receives raw signal data from the Worker and updates the graph."""
        self.raw_signal_plot_item.setData(y=y_data)
        if self.worker:
            # Update labels related to the movement detection buffer
            self.movement_info_label.setText(f"Movement Buffer: {len(self.worker.movement_buffer)}/{self.worker.BUFFER_SIZE}")
            if self.worker.last_peak_index != -1:
                self.peak_index_label.setText(f"Peak Index: {self.worker.last_peak_index}")
            else:
                self.peak_index_label.setText("Peak Index: N/A")

    @pyqtSlot(float)
    def handle_activity_detection(self, trigger_value):
        """Slot for when the Worker detects general activity (peak shift)."""
        self.last_activity_time = time.time()
        self.activity_indicator_label.setText(f"ACTIVITY DETECTED\n(Movement)")
        self.activity_indicator_label.setStyleSheet("background-color: #007bff; color: black; border-radius: 4px; font-weight: bold; font-size: 16px;")
        
        # Turn on the LED to indicate activity (if not already on for a confirmed human)
        if self.led_state != "HUMAN":
            self.led_state = "ACTIVITY"
            try: self.threadpool.start(GenericRunnable(self.rp_sensor.give_ssh_command, "/opt/redpitaya/bin/monitor 0x40000030 0x80"))
            except Exception as e: print(f"Failed to turn on LED on activity: {e}")
            self.update_led_indicator()

    @pyqtSlot(str)
    def handle_mode_change(self, mode):
        """Updates the UI when the Worker changes detection mode."""
        if mode == 'cnn':
            self.mode_label.setText("Mode: CNN Classify")
            self.mode_label.setStyleSheet("font-weight: bold; color: #e67e22; font-size: 14px;")
            self.last_cnn_mode_time = time.time() # Reset inactivity timer for CNN mode
        elif mode == 'peak':
            self.mode_label.setText("Mode: Activity Detection")
            self.mode_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 14px;")

    @pyqtSlot(float)
    def handle_prediction(self, probability):
        """Handles a new prediction from the CNN."""
        self.probability_label.setText(f"Human Prob: {probability:.4f}")
        
        # Debounce logic: require multiple consecutive high-confidence predictions
        if probability > self.detection_threshold: 
            self.debounce_counter += 1
        else: 
            self.debounce_counter = 0 # Reset counter if confidence drops
        
        self.debounce_label.setText(f"Debounce: {self.debounce_counter}/{self.debounce_hits_required}")
        
        if self.debounce_counter >= self.debounce_hits_required: 
            self.confirm_human_detection()

    def confirm_human_detection(self):
        """Called when the debounce logic confirms a human presence."""
        self.last_human_detection_time = time.time()
        self.led_state = "HUMAN"
        try:
            # Run the SSH command in the threadpool to avoid blocking the UI
            self.threadpool.start(GenericRunnable(self.rp_sensor.give_ssh_command, "/opt/redpitaya/bin/monitor 0x40000030 0x80"))
        except Exception as e:
            print(f"Failed to turn on LED for human: {e}")
        self.update_led_indicator()

    def update_led_indicator(self):
        """Updates the LED indicator label in the UI based on the current led_state."""
        if self.led_state == "OFF":
            self.led_indicator.setText("LED: OFF")
            self.led_indicator.setStyleSheet("background-color: #444; color: white; border-radius: 4px; font-weight: bold; font-size: 14px;")
        elif self.led_state == "ACTIVITY":
            self.led_indicator.setText("LED: ON (Activity)")
            self.led_indicator.setStyleSheet("background-color: #FFFFFF; color: black; border-radius: 4px; font-weight: bold; font-size: 14px;")
        elif self.led_state == "HUMAN":
            self.led_indicator.setText("LED: ON (Human)")
            self.led_indicator.setStyleSheet("background-color: #FFFFFF; color: black; border-radius: 4px; font-weight: bold; font-size: 14px;")

    @pyqtSlot()
    def update_human_detection_status(self):
        """Periodically called by a QTimer to manage state timeouts."""
        # Check if a human detection is still considered "current"
        is_human_confirmed = (self.last_human_detection_time > 0 and time.time() - self.last_human_detection_time <= self.detection_timeout_s)
        
        # Check if any activity is still "current"
        is_activity_recent = (self.last_activity_time > 0 and time.time() - self.last_activity_time <= self.detection_timeout_s)

        # Update the main status button
        if is_human_confirmed:
            self.human_detection_status_btn.setText("HUMAN DETECTED")
            self.human_detection_status_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 18px;")
            self.last_cnn_mode_time = time.time() # Keep CNN mode active while human is present
        else:
            self.human_detection_status_btn.setText("NO HUMAN DETECTED")
            self.human_detection_status_btn.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 18px;")
            
            # If human detection timed out, update LED state
            if self.led_state == "HUMAN":
                self.led_state = "OFF" if not is_activity_recent else "ACTIVITY"
                try:
                    if self.led_state == "OFF": self.rp_sensor.give_ssh_command("/opt/redpitaya/bin/monitor 0x40000030 0x0")
                    else: self.rp_sensor.give_ssh_command("/opt/redpitaya/bin/monitor 0x40000030 0x80")
                except Exception as e: print(f"Failed to update LED after human timeout: {e}")
                self.update_led_indicator()

        # Update the activity label
        if not is_activity_recent:
            self.activity_indicator_label.setText("ACTIVITY MODE - IDLE")
            self.activity_indicator_label.setStyleSheet("background-color: #555; border-radius: 4px; font-weight: bold; font-size: 16px;")
            
            # If activity has timed out, turn off the LED (if it's not on for a human)
            if self.led_state != "HUMAN" and self.led_state != "OFF":
                self.led_state = "OFF"
                try: self.rp_sensor.give_ssh_command("/opt/redpitaya/bin/monitor 0x40000030 0x0")
                except Exception as e: print(f"Failed to turn off LED: {e}")
                self.update_led_indicator()

        # If in CNN mode for too long without new detections, revert to peak mode
        if self.worker and self.worker.detection_mode == 'cnn' and not self.worker.cnn_only_mode:
            # Prolonged time out of CNN detection is 4 times the user specified detection timeout
            prolonged_timeout = 4 * self.detection_timeout_s
            if time.time() - self.last_cnn_mode_time > prolonged_timeout:
                self.update_app_status("Inactivity timeout. Reverting to activity detection.", False)
                self.worker.set_detection_mode('peak')

    @pyqtSlot(str, bool)
    def update_app_status(self, msg, is_error):
        """Updates the main application status label at the bottom."""
        self.app_status_label.setText(f"App: {msg}")
        self.app_status_label.setStyleSheet("color: #ff4757;" if is_error else "color: white;")
        
    def update_app_status_conditionally(self, msg, is_error):
        """Only updates the app status if the stream is not yet stable (to avoid spamming)."""
        if not self.stream_is_stable: self.update_app_status(msg, is_error)
        
    @pyqtSlot(str)
    def update_signal_color(self, color_name):
        """Updates the plot line color based on the combobox selection."""
        self.raw_signal_plot_item.setPen(self.color_map.get(color_name, 'w'))
        
    def update_debounce_setting(self, value):
        """Updates the debounce requirement and resets the counter."""
        self.debounce_hits_required, self.debounce_counter = value, 0
        self.debounce_label.setText(f"Debounce: 0/{self.debounce_hits_required}")
        
    @pyqtSlot()
    def update_datetime(self):
        """Updates the time display."""
        self.datetime_label.setText(f"Time: {time.strftime('%H:%M:%S')}")
        
    def closeEvent(self, event):
        """Ensures a clean shutdown when the application window is closed."""
        if self.worker: self.worker.stop()
        # Run the stop sequence in the threadpool and wait for it to finish
        self.threadpool.start(GenericRunnable(self._stop_sensor_sequence))
        self.threadpool.waitForDone()
        event.accept()

# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setStyleSheet(DARK_STYLE)
    window.show()
    sys.exit(app.exec())