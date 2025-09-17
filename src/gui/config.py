# ==============================================================================
#
# Configuration and Styling Constants
#
# This file holds all static configuration values, such as signal
# processing parameters and the UI's visual stylesheet. This makes it easy
# to adjust application-wide settings from a single location.
#
# ==============================================================================

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