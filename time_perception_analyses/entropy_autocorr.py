import numpy as np
import nolds
import pandas as pd

output_df = pd.read_excel("control_time_output.xlsx")

# Initialize dictionaries to store the results
entropy_values = {}
acorr_values = {}

# Define the truncation length
truncation_length = 40000 #3614

# Loop through each column and compute metrics after truncation
for column in output_df.columns:
    # Drop NA values and truncate the series to the first 2000 data points
    clean_series = output_df[column].dropna()
    if len(clean_series) >= truncation_length:
        truncated_series = clean_series[:truncation_length]
    else:
        # If there are fewer than 2000 data points, you could either skip this series or keep it as is
        print(f"Warning: {column} has less than {truncation_length} data points; its results may be unreliable.")
        truncated_series = clean_series  # You could opt to use it as is or skip

    # Compute Sample Entropy
    entropy = nolds.sampen(truncated_series) if len(truncated_series) > 1 else np.nan

    # Compute Autocorrelation for lag 1
    acorr = np.corrcoef(np.array([truncated_series[:-1], truncated_series[1:]]))[0, 1] if len(truncated_series) > 1 else np.nan

    # Store the results
    entropy_values[column] = entropy
    acorr_values[column] = acorr

# Create a DataFrame for the results
entropy_df = pd.DataFrame({
    'Corpora': list(entropy_values.keys()),
    'SampEn': list(entropy_values.values()),
    'Autocorrelation': list(acorr_values.values())
})

# Output the results
print(entropy_df)