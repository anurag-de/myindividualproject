# ==============================================================================
#
# Main Application Window Layout
#
# This module defines the static layout of the main window. It is
# responsible for instantiating and arranging all the graphical widgets
# (buttons, labels, plots, etc.) without assigning any functional logic to them.
# This separates the visual design from the application's behavior.
#
# ==============================================================================

# --- GUI and Plotting Libraries (PyQt6 and pyqtgraph) ---
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QPushButton, QGridLayout, QHBoxLayout,
                             QVBoxLayout, QLabel, QGroupBox, QSpinBox,
                             QCheckBox, QDoubleSpinBox, QComboBox, QFrame)

class Ui_MainWindow:
    """Defines the user interface layout for the main application window."""
    def setupUi(self, MainWindow):
        """Creates and arranges all widgets within the provided QMainWindow."""
        MainWindow.setObjectName("MainWindow")
        
        main_widget = QWidget()
        MainWindow.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Plot Widget ---
        MainWindow.plot_widget = pg.PlotWidget(title="Raw Signal (ADC)")
        MainWindow.plot_widget.showGrid(x=True, y=True)
        MainWindow.raw_signal_plot_item = MainWindow.plot_widget.plot(pen='w')
        main_layout.addWidget(MainWindow.plot_widget)

        bottom_deck_layout = QHBoxLayout()

        # --- Primary Status Group ---
        primary_status_group = QGroupBox("Primary Status")
        primary_status_layout = QVBoxLayout()
        MainWindow.human_detection_status_btn = QPushButton("NO HUMAN DETECTED")
        MainWindow.human_detection_status_btn.setMinimumHeight(50)
        MainWindow.human_detection_status_btn.setStyleSheet("background-color: red; color: white; font-weight: bold; font-size: 18px;")
        MainWindow.activity_indicator_label = QLabel("ACTIVITY MODE - IDLE")
        MainWindow.activity_indicator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        MainWindow.activity_indicator_label.setMinimumHeight(40)
        MainWindow.activity_indicator_label.setStyleSheet("background-color: #555; border-radius: 4px; font-weight: bold; font-size: 16px;")
        MainWindow.mode_label = QLabel("Mode: Activity Detection")
        MainWindow.mode_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 14px;")
        MainWindow.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        MainWindow.led_indicator = QLabel("LED: OFF")
        MainWindow.led_indicator.setStyleSheet("background-color: #444; color: white; border-radius: 4px; font-weight: bold; font-size: 14px;")
        MainWindow.led_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        MainWindow.led_indicator.setMinimumHeight(30)
        primary_status_layout.addWidget(MainWindow.human_detection_status_btn)
        primary_status_layout.addWidget(MainWindow.activity_indicator_label)
        primary_status_layout.addWidget(MainWindow.mode_label)
        primary_status_layout.addWidget(MainWindow.led_indicator)
        primary_status_group.setLayout(primary_status_layout)
        bottom_deck_layout.addWidget(primary_status_group, 2)

        # --- Controls & Settings Group ---
        controls_and_settings_group = QGroupBox("Settings")
        controls_and_settings_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()
        MainWindow.start_sensor_btn = QPushButton("Start Sensor")
        MainWindow.detection_chkbox = QCheckBox("Enable Detection")
        MainWindow.detection_chkbox.setEnabled(False)
        MainWindow.stop_sensor_btn = QPushButton("Stop Sensor")
        MainWindow.stop_sensor_btn.setEnabled(False)
        controls_layout.addWidget(MainWindow.start_sensor_btn)
        controls_layout.addWidget(MainWindow.detection_chkbox)
        controls_layout.addWidget(MainWindow.stop_sensor_btn)
        controls_and_settings_layout.addLayout(controls_layout)
        MainWindow.detection_mode_btn = QPushButton("Switch to CNN-only Mode")
        MainWindow.detection_mode_btn.setEnabled(False)
        controls_and_settings_layout.addWidget(MainWindow.detection_mode_btn)
        controls_and_settings_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine))
        settings_layout = QGridLayout()
        settings_layout.addWidget(QLabel("Timeout (s):"), 0, 0)
        MainWindow.timeout_spinbox = QSpinBox()
        MainWindow.timeout_spinbox.setRange(1, 120)
        settings_layout.addWidget(MainWindow.timeout_spinbox, 0, 1)
        settings_layout.addWidget(QLabel("Confidence:"), 1, 0)
        MainWindow.prob_spinbox = QDoubleSpinBox()
        MainWindow.prob_spinbox.setRange(0.5, 1.0)
        MainWindow.prob_spinbox.setSingleStep(0.01)
        settings_layout.addWidget(MainWindow.prob_spinbox, 1, 1)
        settings_layout.addWidget(QLabel("Debounce:"), 0, 2)
        MainWindow.debounce_spinbox = QSpinBox()
        MainWindow.debounce_spinbox.setRange(1, 20)
        settings_layout.addWidget(MainWindow.debounce_spinbox, 0, 3)
        settings_layout.addWidget(QLabel("Color:"), 1, 2)
        MainWindow.color_combobox = QComboBox()
        settings_layout.addWidget(MainWindow.color_combobox, 1, 3)
        settings_layout.addWidget(QLabel("Movement Thresh (idx):"), 2, 0)
        MainWindow.movement_threshold_spinbox = QSpinBox()
        MainWindow.movement_threshold_spinbox.setRange(50, 5000)
        MainWindow.movement_threshold_spinbox.setSingleStep(50)
        MainWindow.movement_threshold_spinbox.setValue(2000)
        settings_layout.addWidget(MainWindow.movement_threshold_spinbox, 2, 1)
        controls_and_settings_layout.addLayout(settings_layout)
        controls_and_settings_group.setLayout(controls_and_settings_layout)
        bottom_deck_layout.addWidget(controls_and_settings_group, 3)

        # --- Detailed Status Group ---
        status_groupbox = QGroupBox("Detailed Status")
        status_layout = QGridLayout()
        MainWindow.app_status_label = QLabel("App: Initializing, please wait...")
        MainWindow.sensor_status_label = QLabel("Sensor: Waiting")
        MainWindow.stream_status_label = QLabel("Stream: Idle")
        MainWindow.datetime_label = QLabel("Time: N/A")
        MainWindow.probability_label = QLabel("Human Prob: N/A")
        MainWindow.debounce_label = QLabel("Debounce: 0/3")
        MainWindow.total_signals_label = QLabel("Complete: 0")
        MainWindow.broken_signals_label = QLabel("Broken: 0")
        MainWindow.movement_info_label = QLabel("Movement Buffer: 0/4")
        MainWindow.peak_index_label = QLabel("Peak Index: N/A")
        status_layout.addWidget(MainWindow.app_status_label, 0, 0, 1, 2)
        status_layout.addWidget(MainWindow.sensor_status_label, 1, 0, 1, 2)
        status_layout.addWidget(MainWindow.stream_status_label, 2, 0)
        status_layout.addWidget(MainWindow.datetime_label, 2, 1)
        status_layout.addWidget(QFrame(frameShape=QFrame.Shape.HLine), 3, 0, 1, 2)
        status_layout.addWidget(MainWindow.probability_label, 4, 0)
        status_layout.addWidget(MainWindow.debounce_label, 4, 1)
        status_layout.addWidget(MainWindow.total_signals_label, 5, 0)
        status_layout.addWidget(MainWindow.broken_signals_label, 5, 1)
        status_layout.addWidget(MainWindow.movement_info_label, 6, 0)
        status_layout.addWidget(MainWindow.peak_index_label, 6, 1)
        status_groupbox.setLayout(status_layout)
        bottom_deck_layout.addWidget(status_groupbox, 2)

        main_layout.addLayout(bottom_deck_layout)
        main_layout.setStretch(0, 5)
        main_layout.setStretch(1, 5)