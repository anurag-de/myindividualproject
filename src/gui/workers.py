# ==============================================================================
#
# Worker Threads
#
# This module contains all QRunnable classes that perform background
# tasks. This includes the StartupWorker for initializing the sensor connection,
# the main Worker for continuous data acquisition and processing, and a generic
# runnable for simple, one-off tasks. Using workers ensures the GUI remains
# responsive during long-running operations.
#
# ==============================================================================

# --- Core Python and System Libraries ---
import time

# --- GUI and Plotting Libraries (PyQt6) ---
from PyQt6.QtCore import QRunnable, pyqtSlot, QObject, pyqtSignal

# --- Machine Learning and Signal Processing Libraries ---
import numpy as np
import torch
from scipy.signal import find_peaks

# --- Import from local modules ---
from ml_model import preprocess_signal_for_pytorch
from config import FIXED_LENGTH

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
        activity_detected = pyqtSignal()
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
                    self.signals.activity_detected.emit() # Emit activity signal
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