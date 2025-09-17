# ==============================================================================
#
# PyTorch Model Definition & Preprocessing
#
# This module defines the architecture of the 2D Convolutional
# Neural Network (CNN) used for signal classification. It also contains the
# necessary function to preprocess a raw 1D signal into a 2D spectrogram
# tensor that the model can accept as input.
#
# ==============================================================================

# --- Machine Learning and Signal Processing Libraries ---
import numpy as np
import torch
import torch.nn as nn

# --- Suppress common warnings for a cleaner console output ---
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- Import from local modules ---
from config import FIXED_LENGTH, N_FFT, HOP_LENGTH

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