# ==============================================================================
#
# Main Application Window Logic
#
# This module implements the core logic and event handling for the
# main application window. It uses the UI layout defined in 'ui_layout.py'
# and connects it to the background workers and sensor communication layers.
# This file is responsible for state management and user interaction responses.
#
# ==============================================================================

# --- Core Python and System Libraries ---
import time

# --- GUI and Plotting Libraries (PyQt6) ---
from PyQt6.QtCore import QThreadPool, QTimer, pyqtSlot
from PyQt6.QtWidgets import QMainWindow

# --- Machine Learning and Signal Processing Libraries ---
import torch
import numpy as np

# --- Import from local modules ---
from ui.ui_layout import Ui_MainWindow
from sensor_comm import RedPitayaSensor
from workers import Worker, StartupWorker, GenericRunnable
from ml_model import SignalCNN2D
from app_state import ApplicationState
from config import N_FFT, HOP_LENGTH, FIXED_LENGTH

# ==============================================================================
# MAIN APPLICATION WINDOW
# ==============================================================================
class MainWindow(QMainWindow):
    """
    The main application window, responsible for orchestrating the UI,
    state manager, and worker threads.
    """
    def __init__(self):
        super().__init__()
        # --- Core Components ---
        self.rp_sensor = RedPitayaSensor()
        self.threadpool = QThreadPool()
        self.state = ApplicationState()
        self.worker, self.model = None, None
        
        self.color_map = {'White': 'w', 'Red': 'r', 'Green': 'g', 'Blue': 'b', 'Yellow': 'y', 'Cyan': 'c', 'Magenta': 'm'}
        
        # --- Initialization ---
        self.setWindowTitle("Human Presence Detector (using PyTorch CNN)")
        self.resize(1200, 650)
        
        # Build the UI from the layout file
        ui = Ui_MainWindow()
        ui.setupUi(self)

        # Set initial values, connect signals, setup model and timers
        self._initialize_settings()
        self._connect_signals()
        self.setup_model()
        self.setup_timers()

    def _initialize_settings(self):
        """Sets the initial values for the settings widgets from the state."""
        self.timeout_spinbox.setValue(self.state.detection_timeout_s)
        self.prob_spinbox.setValue(self.state.detection_threshold)
        self.debounce_spinbox.setValue(self.state.debounce_hits_required)
        self.color_combobox.addItems(self.color_map.keys())

    def _connect_signals(self):
        """Connects widget, worker, and state signals to their slots."""
        # UI Widget Signals
        self.start_sensor_btn.clicked.connect(self.start_sensor_btn_handler)
        self.stop_sensor_btn.clicked.connect(self.stop_sensor_btn_handler)
        self.detection_chkbox.toggled.connect(self.detection_checkbox_handler)
        self.detection_mode_btn.clicked.connect(self.toggle_detection_mode)
        
        # Settings Signals
        self.timeout_spinbox.valueChanged.connect(lambda v: setattr(self.state, 'detection_timeout_s', v))
        self.prob_spinbox.valueChanged.connect(lambda v: setattr(self.state, 'detection_threshold', v))
        self.debounce_spinbox.valueChanged.connect(self.update_debounce_setting)
        self.color_combobox.currentTextChanged.connect(self.update_signal_color)
        self.movement_threshold_spinbox.valueChanged.connect(
            lambda v: self.worker and setattr(self.worker, 'movement_index_threshold', v)
        )
        
        # State Manager Signal
        self.state.state_changed.connect(self.on_state_updated)

    def setup_model(self):
        """Loads the pre-trained PyTorch model from disk."""
        try:
            spec_h = (N_FFT // 2) + 1
            spec_w = (FIXED_LENGTH // HOP_LENGTH) + 1
            self.model = SignalCNN2D((1, spec_h, spec_w))
            model_path = 'myfullmodel/best_model.pth'
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            self.app_status_label.setText(f"App: PyTorch model loaded from '{model_path}'.")
        except Exception as e: 
            self.app_status_label.setText(f"App: Error loading model: {e}")
            self.app_status_label.setStyleSheet("color: #ff4757;")
            self.model = None

    def setup_timers(self):
        """Initializes QTimers for periodic UI and state updates."""
        # Timer to check for timeouts in the state manager
        self.logic_timer = QTimer(self)
        self.logic_timer.setInterval(1000)
        self.logic_timer.timeout.connect(self.run_state_timeout_check)
        self.logic_timer.start()

        # Timers for pure UI updates
        self.status_message_timer = QTimer(self)
        self.status_message_timer.setInterval(250)
        self.status_message_timer.timeout.connect(lambda: self.sensor_status_label.setText(f"Sensor: {self.rp_sensor.get_sensor_status_message()}"))
        self.status_message_timer.start()

        self.datetime_timer = QTimer(self)
        self.datetime_timer.setInterval(1000)
        self.datetime_timer.timeout.connect(lambda: self.datetime_label.setText(f"Time: {time.strftime('%H:%M:%S')}"))
        self.datetime_timer.start()
        
    def start_sensor_btn_handler(self):
        """Handles the 'Start Sensor' button click."""
        self.start_sensor_btn.setEnabled(False)
        self.stop_sensor_btn.setEnabled(True)
        self.state.reset()
        
        startup_worker = StartupWorker(self.rp_sensor)
        startup_worker.signals.status_updated.connect(self.update_app_status_conditionally)
        startup_worker.signals.startup_successful.connect(self.on_startup_successful)
        startup_worker.signals.startup_failed.connect(self.on_startup_failed)
        self.threadpool.start(startup_worker)

    @pyqtSlot()
    def on_startup_successful(self):
        """Launches the main worker after successful sensor startup."""
        self.update_app_status_conditionally("Starting data acquisition...", False)
        if not self.model:
            self.update_app_status("Cannot start: PyTorch model not loaded.", True)
            return
            
        self.worker = Worker(self.rp_sensor, self.threadpool, self.model)
        self.worker.movement_index_threshold = self.movement_threshold_spinbox.value()
        
        # Connect worker signals
        self.worker.signals.raw_plot_ready.connect(self.update_raw_plot)
        self.worker.signals.prediction_made.connect(self.state.process_new_prediction)
        self.worker.signals.status_updated.connect(self.update_app_status)
        self.worker.signals.increment_total.connect(self.state.increment_total_signals)
        self.worker.signals.increment_broken.connect(self.state.increment_broken_signals)
        self.worker.signals.activity_detected.connect(self.state.report_activity)
        self.worker.signals.mode_changed.connect(self.handle_mode_change)
        
        self.threadpool.start(self.worker)
        self.detection_chkbox.setEnabled(True)
        self.detection_mode_btn.setEnabled(True)

    def stop_sensor_btn_handler(self):
        """Handles the 'Stop Sensor' button click."""
        self.detection_chkbox.setChecked(False)
        self.detection_chkbox.setEnabled(False)
        self.detection_mode_btn.setEnabled(False)
        if self.worker: self.worker.stop()
        
        self.threadpool.start(GenericRunnable(self._stop_sensor_sequence))
        
        self.update_app_status("Stop command issued.", False)
        self.rp_sensor.set_sensor_message("Sensor stopped.")
        self.stream_status_label.setText("Stream: Idle"); self.stream_status_label.setStyleSheet("color: white;")
        self.start_sensor_btn.setEnabled(True)
        self.stop_sensor_btn.setEnabled(False)
        self._update_led_display("OFF")
        self._update_remote_led("OFF")

    @pyqtSlot(dict)
    def on_state_updated(self, payload):
        """Central slot to update the UI whenever the application state changes."""
        if payload.get('full_reset', False):
            self.app_status_label.setText("App: Waiting for stable stream...")
            self.stream_status_label.setText("Stream: Unstable"); self.stream_status_label.setStyleSheet("color: #ff4757;")
            self.total_signals_label.setText("Complete: 0")
            self.broken_signals_label.setText("Broken: 0")
            self.raw_signal_plot_item.clear()
            self.probability_label.setText("Human Prob: N/A")
            self.movement_info_label.setText("Movement Buffer: 0/4")
            self.peak_index_label.setText("Peak Index: N/A")
            self.update_human_detection_display(False)
            self.update_activity_display(False)
            self._update_led_display("OFF")

        if 'debounce_text' in payload: self.debounce_label.setText(payload['debounce_text'])
        if 'probability_text' in payload: self.probability_label.setText(payload['probability_text'])
        if 'total_signals_text' in payload: self.total_signals_label.setText(payload['total_signals_text'])
        if 'broken_signals_text' in payload: self.broken_signals_label.setText(payload['broken_signals_text'])
        
        if 'stream_stable' in payload:
            is_stable = payload['stream_stable']
            self.update_app_status("Sensor stream is stable." if is_stable else "Stream unstable!", not is_stable)
            self.stream_status_label.setText("Stream: Stable" if is_stable else "Stream: Unstable")
            self.stream_status_label.setStyleSheet("color: #2ecc71;" if is_stable else "color: #ff4757;")

        if 'activity_detected' in payload: self.update_activity_display(True)
        if 'activity_timed_out' in payload: self.update_activity_display(False)
        
        if 'human_detected' in payload: self.update_human_detection_display(payload['human_detected'])
        
        if 'led_state' in payload:
            self._update_led_display(payload['led_state'])
            self._update_remote_led(payload['led_state'])

        if payload.get('revert_to_peak_mode', False) and self.worker:
            self.update_app_status("Inactivity timeout. Reverting to activity detection.", False)
            self.worker.set_detection_mode('peak')

    def run_state_timeout_check(self):
        """Called by a timer to trigger the state manager's timeout checks."""
        if self.worker:
            self.state.check_timeouts(
                self.worker.detection_mode == 'cnn', 
                self.worker.cnn_only_mode
            )

    def update_human_detection_display(self, detected):
        """Updates the main status button."""
        if detected:
            self.human_detection_status_btn.setText("HUMAN DETECTED")
            self.human_detection_status_btn.setStyleSheet("background-color: green; color: white; font-weight: bold; font-size: 18px;")
        else:
            self.human_detection_status_btn.setText("NO HUMAN DETECTED")
            self.human_detection_status_btn.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 18px;")
            
    def update_activity_display(self, detected):
        """Updates the activity indicator label."""
        if detected:
            self.activity_indicator_label.setText(f"ACTIVITY DETECTED\n(Movement)")
            self.activity_indicator_label.setStyleSheet("background-color: #007bff; color: black; border-radius: 4px; font-weight: bold; font-size: 16px;")
        else:
            self.activity_indicator_label.setText("ACTIVITY MODE - IDLE")
            self.activity_indicator_label.setStyleSheet("background-color: #555; border-radius: 4px; font-weight: bold; font-size: 16px;")

    def _update_led_display(self, state):
        """Updates the LED indicator label in the UI."""
        text = {"OFF": "LED: OFF", "ACTIVITY": "LED: ON (Activity)", "HUMAN": "LED: ON (Human)"}
        style = {"OFF": "background-color: #444; color: white;", "ACTIVITY": "background-color: #FFFFFF; color: black;", "HUMAN": "background-color: #FFFFFF; color: black;"}
        base = "border-radius: 4px; font-weight: bold; font-size: 14px;"
        self.led_indicator.setText(text.get(state, "LED: UNKNOWN"))
        self.led_indicator.setStyleSheet(style.get(state, "") + base)

    def _update_remote_led(self, state):
        """Sends the SSH command to update the physical LED on the device."""
        command = "/opt/redpitaya/bin/monitor 0x40000030 0x0" # Default OFF
        if state in ["ACTIVITY", "HUMAN"]:
            command = "/opt/redpitaya/bin/monitor 0x40000030 0x80" # ON
        self.threadpool.start(GenericRunnable(self.rp_sensor.give_ssh_command, command))
    
    # --- Passthrough methods and other slots ---

    def _stop_sensor_sequence(self):
        try:
            if pid := self.rp_sensor.give_ssh_command("pidof dma_with_udp_faster").strip(): 
                self.rp_sensor.give_ssh_command(f"kill {pid}")
            self.rp_sensor.give_ssh_command("/opt/redpitaya/bin/monitor 0x40000030 0x0")
        finally: 
            self.rp_sensor.disconnect_ssh()

    @pyqtSlot(bool)
    def detection_checkbox_handler(self, state):
        if self.worker: self.worker.set_detection_enabled(state)

    def toggle_detection_mode(self):
        if not self.worker: return
        self.worker.set_cnn_only_mode(not self.worker.cnn_only_mode)
        text = "Switch to Activity+CNN Mode" if self.worker.cnn_only_mode else "Switch to CNN-only Mode"
        self.detection_mode_btn.setText(text)

    @pyqtSlot(str)
    def handle_mode_change(self, mode):
        if mode == 'cnn':
            self.mode_label.setText("Mode: CNN Classify")
            self.mode_label.setStyleSheet("font-weight: bold; color: #e67e22; font-size: 14px;")
            self.state.last_cnn_mode_time = time.time()
        else: # mode == 'peak'
            self.mode_label.setText("Mode: Activity Detection")
            self.mode_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 14px;")

    @pyqtSlot(np.ndarray)
    def update_raw_plot(self, y_data):
        self.raw_signal_plot_item.setData(y=y_data)
        if self.worker:
            self.movement_info_label.setText(f"Movement Buffer: {len(self.worker.movement_buffer)}/{self.worker.BUFFER_SIZE}")
            self.peak_index_label.setText(f"Peak Index: {self.worker.last_peak_index}" if self.worker.last_peak_index != -1 else "Peak Index: N/A")

    @pyqtSlot(str)
    def update_signal_color(self, color_name):
        self.raw_signal_plot_item.setPen(self.color_map.get(color_name, 'w'))

    def update_debounce_setting(self, value):
        self.state.debounce_hits_required = value
        self.state.debounce_counter = 0
        self.debounce_label.setText(f"Debounce: 0/{value}")

    @pyqtSlot(str, bool)
    def update_app_status(self, msg, is_error):
        self.app_status_label.setText(f"App: {msg}")
        self.app_status_label.setStyleSheet("color: #ff4757;" if is_error else "color: white;")
        
    def update_app_status_conditionally(self, msg, is_error):
        if not self.state.stream_is_stable: self.update_app_status(msg, is_error)

    @pyqtSlot(str)
    def on_startup_failed(self, error_message):
        self.update_app_status(error_message, True)
        self.start_sensor_btn.setEnabled(True)
        self.stop_sensor_btn.setEnabled(False)

    def closeEvent(self, event):
        if self.worker: self.worker.stop()
        self.threadpool.start(GenericRunnable(self._stop_sensor_sequence))
        self.threadpool.waitForDone()
        event.accept()