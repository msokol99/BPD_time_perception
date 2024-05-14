import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_data_dir = 'data/train_data.csv'
val_data_dir = 'data/val_data.csv'

# Load the ANPST 718 dataset (Imbir, 2016)
df = pd.read_excel('C:/Users/marta/Desktop/Dane/Training Data/Annotated datasets/ANPST_718_Dataset.xlsx')

# Load the annotated dataset
df2 = pd.read_csv('C:/Users/marta/Desktop/Magisterka/Anotacje instrukcje itp/bpd_training_dataset.csv')

def preprocess_data(df, df2):
    """
    Preprocesses the input dataframes by cleaning, merging, and categorizing them.

    Args:
        df (DataFrame): The first dataframe to be processed.
        df2 (DataFrame): The second dataframe to be processed.

    Returns:
        DataFrame: The preprocessed and concatenated dataframe.
    """
    # Remove specific unnamed columns and the first row which might be headers or indexes
    df.drop(columns=['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.drop(labels=0, inplace=True)

    # Keep only the first three columns assuming they correspond to 'sentence', 'valence', and 'arousal'
    df = df.iloc[:, :3].copy()

    # Rename columns for clarity
    df.columns = ['sentence', 'valence', 'arousal']

    # Concatenate both dataframes into a single dataframe for processing
    df_concat = pd.concat([df, df2], ignore_index=True)

    # Drop any rows with NaN values to ensure data quality
    df_concat.dropna(axis=0, inplace=True)

    # Compute the quantile-based bins for the 'arousal' column using 3 classes
    num_bins = 3
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(df_concat['arousal'], quantiles)

    # Assign each 'arousal' value to a bin/category
    df_concat['arousal'] = pd.cut(df_concat['arousal'], bins=bin_edges, labels=[0, 1, 2], include_lowest=True)

    # Drop the 'valence' column if it is not needed for further analysis
    df_concat.drop(columns=['valence'], inplace=True)

    return df_concat

# Split the preprocessed data into training and validation sets
train_df, val_df = train_test_split(preprocess_data(df, df2), test_size=0.2, random_state=42, shuffle=True)

# Save the processed datasets to CSV files for later use
train_df.to_csv(train_data_dir, encoding='utf-8-sig', index=False)
val_df.to_csv(val_data_dir, encoding='utf-8-sig', index=False)
