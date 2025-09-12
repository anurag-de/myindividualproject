################################################################################
# Author:      Anurag
#
# Description: This script preprocesses raw signal data from CSV files.
#              It loads signals, standardizes their length, computes their
#              spectrograms using a Short-Time Fourier Transform (STFT),
#              and saves the resulting spectrograms and their corresponding
#              labels into NumPy (.npy) files.
################################################################################

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def process_and_save_spectrograms(data_path, output_path):
    """
    Processes a directory of raw signals, converts them to spectrograms,
    and saves them as a NumPy dataset.

    Args:
        data_path (str): The path to the root directory containing the data.
                         This directory should have 'human' and 'nonhuman' subfolders.
        output_path (str): The directory where the output .npy files will be saved.
    """
    print(f"--- Starting preprocessing for: {os.path.basename(data_path)} ---")
    
    # --- FIXED PARAMETERS ---
    # Define a fixed length for all signals to ensure uniform input size.
    FIXED_LENGTH = 19517  # This corresponds to 25017 - 5500 from the original logic.
    
    # STFT (Short-Time Fourier Transform) parameters for spectrogram generation.
    n_fft, hop_length = 256, 128
    window = torch.hann_window(n_fft) # Use a Hann window for STFT.
    
    # Lists to store all processed data and labels before saving.
    all_spectrograms = []
    all_labels = []

    # Iterate over the two main categories: 'human' (label 1) and 'nonhuman' (label 0).
    for category, label in [('human', 1), ('nonhuman', 0)]:
        category_path = os.path.join(data_path, category)
        # Check if the category directory exists before proceeding.
        if not os.path.isdir(category_path):
            print(f"Warning: Directory not found for '{category}' in {data_path}. Skipping.")
            continue
            
        # List all subdirectories (scenarios) within the category folder.
        scenarios = [s for s in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, s))]
        
        print(f"-> Found {len(scenarios)} scenarios for '{category}' category.")
        # Process each scenario using a progress bar (tqdm).
        for scenario in tqdm(scenarios, desc=f"Processing '{category}' scenarios"):
            scenario_path = os.path.join(category_path, scenario)
            # Find the first CSV file in the scenario directory.
            csv_file = next((f for f in os.listdir(scenario_path) if f.endswith('.csv')), None)
            if not csv_file: continue

            # Load the signal data from the CSV file.
            df = pd.read_csv(os.path.join(scenario_path, csv_file), header=None)
            if df.empty:
                print(f"Warning: Skipping empty file in scenario {scenario}")
                continue

            # Each row in the CSV is treated as a separate signal.
            for _, row in df.iterrows():
                # Step 1: Load, slice, and standardize signal length.
                # Skip signals that are too short to be meaningful.
                if row.shape[0] <= 5500:
                    continue
                # Trim the initial part of the signal.
                signal = row.values[5500:].astype(np.float32)

                # Truncate longer signals to FIXED_LENGTH.
                if signal.shape[0] > FIXED_LENGTH:
                    signal = signal[:FIXED_LENGTH]
                # Pad shorter signals with zeros to reach FIXED_LENGTH.
                elif signal.shape[0] < FIXED_LENGTH:
                    padding = np.zeros(FIXED_LENGTH - signal.shape[0], dtype=np.float32)
                    signal = np.concatenate((signal, padding))
                
                # Step 2: Convert the NumPy signal array to a PyTorch Tensor.
                signal_tensor = torch.from_numpy(signal)
                
                # Step 3: Compute the Spectrogram using STFT.
                # The result 'spec' is a complex tensor.
                spec = torch.stft(signal_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
                # We take the absolute value to get the magnitude spectrogram.
                spectrogram = torch.abs(spec)
                
                # Step 4: Append the processed spectrogram and its label to our lists.
                # Convert the tensor back to a NumPy array for storage.
                all_spectrograms.append(spectrogram.numpy())
                all_labels.append(label)

    print("\n-> Stacking and saving data to .npy files...")
    # If no data was found or processed, abort the save operation.
    if not all_spectrograms:
        print("Error: No data was processed. Aborting save.")
        return

    # Convert the lists of arrays into single large NumPy arrays.
    spectrograms_array = np.array(all_spectrograms, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    # Create the output directory if it doesn't already exist.
    os.makedirs(output_path, exist_ok=True)
    
    # Generate filenames based on the input data directory's name.
    dataset_name = os.path.basename(os.path.normpath(data_path))
    np.save(os.path.join(output_path, f'{dataset_name}_spectrograms.npy'), spectrograms_array)
    np.save(os.path.join(output_path, f'{dataset_name}_labels.npy'), labels_array)
    
    # Print a confirmation message with the shapes of the saved arrays.
    print(f"✅ Preprocessing complete for {dataset_name}.")
    print(f"   - Spectrograms saved to: {os.path.join(output_path, f'{dataset_name}_spectrograms.npy')} with shape {spectrograms_array.shape}")
    print(f"   - Labels saved to:     {os.path.join(output_path, f'{dataset_name}_labels.npy')} with shape {labels_array.shape}")


def main():
    """
    Main function to parse command-line arguments and run the preprocessing.
    """
    # Set up argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(description="Convert raw signal data into spectrogram NumPy datasets.")
    parser.add_argument("data_path", type=str, help="Path to the ROOT data directory (e.g., 'data').")
    parser.add_argument("--output_path", type=str, default='./preprocessed_data', help="Directory to save the output .npy files.")
    args = parser.parse_args()
    
    # Call the main processing function with the provided arguments.
    process_and_save_spectrograms(args.data_path, args.output_path)
    
    print("\nDataset processed successfully.")

# Standard Python entry point.
if __name__ == '__main__':
    main()