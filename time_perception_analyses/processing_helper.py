from sum_word_len_sentiment import get_lengths_and_frequencies, calculate_adjustments, process_data, main_analysis, exponential_smoothing, calculate_autocorrelation
import os
import pandas as pd
import numpy as np
from glob import glob
from natsort import natsorted

# Load directories from Excel files (assuming these files contain the relevant paths)
df_raw_dir = pd.read_excel("input_corpora.xlsx")  # Contains paths to raw data directories
df_valence_dir = pd.read_excel('output_corpora/valence.xlsx')  # Contains paths to valence directories
df_arousal_dir = pd.read_excel('output_corpora/arousal.xlsx')  # Contains paths to arousal directories

# Define a function to process directories
def process_directories(raw_dir, valence_dir, arousal_dir):
    # Iterate through each subdirectory ("borderline" and "control")
    for subdir in ['borderline', 'control']:
        # Construct the full path for each subdirectory
        raw_files = natsorted(glob(os.path.join(raw_dir, subdir, "*.xlsx")))
        valence_files = natsorted(glob(os.path.join(valence_dir, subdir, "*.xlsx")))
        arousal_files = natsorted(glob(os.path.join(arousal_dir, subdir, "*.xlsx")))

        # Iterate through files based on matching filenames (assuming filenames are consistent and only differ by their parent directory)
        for raw_file, valence_file, arousal_file in zip(raw_files, valence_files, arousal_files):
            # Extracting file numbers and ensuring they match
            file_num = os.path.splitext(os.path.basename(raw_file))[0].split('file')[-1]
            if all(file_num in file for file in [valence_file, arousal_file]):
                # Load data
                df_raw = pd.read_excel(raw_file)
                df_valence = pd.read_excel(valence_file)
                df_arousal = pd.read_excel(arousal_file)

                # Apply your data processing functions here
                # For instance:
                final_result = main_analysis(df_valence, df_arousal)
                
                # Further processing, visualization, saving results, etc.
                # For example, you might want to plot or analyze the results here

# Process each set of directories
for index, row in df_raw_dir.iterrows():
    raw_dir = row['Path']  # Assuming the column is named 'Path'
    valence_dir = df_valence_dir.loc[index, 'Path']
    arousal_dir = df_arousal_dir.loc[index, 'Path']

    process_directories(raw_dir, valence_dir, arousal_dir)

