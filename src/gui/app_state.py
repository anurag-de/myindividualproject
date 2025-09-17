# ==============================================================================
#
# Application State Manager
#
# This module contains the ApplicationState class, which serves as
# the "brain" or "model" of the application. It holds all state variables
# (like counters and timers) and contains the core logic for processing data
# and managing state transitions. It uses signals to notify the UI of any
# changes, decoupling the logic from the view.
#
# ==============================================================================

import time
from PyQt6.QtCore import QObject, pyqtSignal

class ApplicationState(QObject):
    """Manages the application's data and business logic."""
    
    # Signal that emits a dictionary payload describing what changed.
    state_changed = pyqtSignal(dict)
    
    STABLE_STREAM_THRESHOLD = 5

    def __init__(self):
        super().__init__()
        
        # --- User-configurable Settings ---
        self.detection_timeout_s = 15
        self.detection_threshold = 0.99
        self.debounce_hits_required = 3
        
        # --- Initialize State ---
        self.reset()

    def reset(self):
        """Resets all state variables to their initial values."""
        self.last_human_detection_time = 0
        self.last_activity_time = 0
        self.last_cnn_mode_time = 0
        self.total_signals_count = 0
        self.broken_signals_count = 0
        self.debounce_counter = 0
        self.consecutive_good_signals = 0
        self.stream_is_stable = False
        self.led_state = "OFF"
        
        # Emit a signal to notify the UI that everything has been reset
        self.state_changed.emit({
            'full_reset': True,
            'debounce_text': f"Debounce: 0/{self.debounce_hits_required}"
        })

    def process_new_prediction(self, probability):
        """Handles the logic for a new prediction from the CNN."""
        if probability > self.detection_threshold: 
            self.debounce_counter += 1
        else: 
            self.debounce_counter = 0
        
        self.state_changed.emit({
            'probability_text': f"Human Prob: {probability:.4f}",
            'debounce_text': f"Debounce: {self.debounce_counter}/{self.debounce_hits_required}"
        })
        
        if self.debounce_counter >= self.debounce_hits_required: 
            self._confirm_human_detection()

    def _confirm_human_detection(self):
        """Logic for when a human presence is confirmed by the debouncer."""
        self.last_human_detection_time = time.time()
        self.led_state = "HUMAN"
        # The UI will be responsible for sending the actual SSH command
        self.state_changed.emit({'led_state': self.led_state})

    def report_activity(self):
        """Logic for when general activity is detected."""
        self.last_activity_time = time.time()
        if self.led_state != "HUMAN":
            self.led_state = "ACTIVITY"
        self.state_changed.emit({
            'activity_detected': True,
            'led_state': self.led_state
        })

    def increment_total_signals(self):
        """Increments the count of successfully received signals."""
        self.total_signals_count += 1
        payload = {'total_signals_text': f"Complete: {self.total_signals_count}"}
        
        if not self.stream_is_stable:
            self.consecutive_good_signals += 1
            if self.consecutive_good_signals >= self.STABLE_STREAM_THRESHOLD:
                self.stream_is_stable = True
                payload['stream_stable'] = True
        
        self.state_changed.emit(payload)

    def increment_broken_signals(self):
        """Increments the count of broken signals and resets stability."""
        self.broken_signals_count += 1
        payload = {'broken_signals_text': f"Broken: {self.broken_signals_count}"}

        self.consecutive_good_signals = 0
        if self.stream_is_stable: 
            self.stream_is_stable = False
            payload['stream_stable'] = False
        
        self.state_changed.emit(payload)

    def check_timeouts(self, detection_mode_is_cnn, is_cnn_only_mode):
        """Periodically checks for various state timeouts."""
        now = time.time()
        is_human_confirmed = (self.last_human_detection_time > 0 and now - self.last_human_detection_time <= self.detection_timeout_s)
        is_activity_recent = (self.last_activity_time > 0 and now - self.last_activity_time <= self.detection_timeout_s)

        payload = {'human_detected': is_human_confirmed}

        if is_human_confirmed:
            self.last_cnn_mode_time = now
        elif self.led_state == "HUMAN":
            self.led_state = "ACTIVITY" if is_activity_recent else "OFF"
            payload['led_state'] = self.led_state
        
        if not is_activity_recent:
            payload['activity_timed_out'] = True
            if self.led_state == "ACTIVITY":
                self.led_state = "OFF"
                payload['led_state'] = self.led_state

        if detection_mode_is_cnn and not is_cnn_only_mode:
            prolonged_timeout = 4 * self.detection_timeout_s
            if now - self.last_cnn_mode_time > prolonged_timeout:
                payload['revert_to_peak_mode'] = True

        self.state_changed.emit(payload)