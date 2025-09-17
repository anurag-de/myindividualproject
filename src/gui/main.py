# ==============================================================================
#
# Human Presence Detector using Red Pitaya board, SRF02 ultrasonicsensor & PyTorch CNN v1.0
#
# Author: Anurag
#
# This is the main entry point for the application. It initializes
# the PyQt6 application, creates the main window, applies the stylesheet,
# and starts the event loop.
#
# ==============================================================================

# --- Core Python and System Libraries ---
import sys

# --- GUI and Plotting Libraries (PyQt6) ---
from PyQt6.QtWidgets import QApplication

# --- Import from local modules ---
from ui.ui_main_window import MainWindow
from config import DARK_STYLE

# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setStyleSheet(DARK_STYLE)
    window.show()
    sys.exit(app.exec())