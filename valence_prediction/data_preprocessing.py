import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_data_dir = 'valence_prediction/data/train_data.csv'
val_data_dir = 'valence_prediction/data/val_data.csv'

# Load the ANPST 718 dataset (Imbir, 2016)
df = pd.read_excel('C:/Users/marta/OneDrive/Desktop/Dane/Training Data/Annotated datasets/ANPST_718_Dataset.xlsx')

# Load the annotated dataset
df2 = pd.read_excel('C:/Users/marta/OneDrive/Desktop/Magisterka/Anotacje instrukcje itp/nonannotated_shuffled.xlsx')

def preprocess_data(df, df2):

    df.drop(columns=['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.drop(labels=0, inplace=True)

    df = df.iloc[:, :3].copy()

    df.columns = ['sentence', 'valence', 'arousal']

    # Concatenate the dataframes
    df_concat = pd.concat([df, df2], ignore_index=True)

    # Drop rows with NaN values
    df_concat.dropna(axis=0, inplace=True)

    # Specify the number of bins or quantiles (e.g., 3 for 3 classes)
    num_bins = 3

    # Compute quantiles and bin the data for arousal
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(df_concat['valence'], quantiles)

    # Apply the binning to the 'arousal' column
    df_concat['valence'] = pd.cut(df_concat['valence'], bins=bin_edges, labels=[0, 1, 2], include_lowest=True)

    # If 'valence' column is not needed, drop it
    df_concat.drop(columns=['arousal'], inplace=True)

    return df_concat


# Train-Test-Validation Split
train_df, val_df = train_test_split(preprocess_data(df), test_size=0.2, random_state=42)

# Save the preprocessed data to CSV files
train_df.to_csv(train_data_dir, encoding='utf-8-sig', index=False)
val_df.to_csv(val_data_dir, encoding='utf-8-sig', index=False)