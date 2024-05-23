import numpy as np
import nolds
import pandas as pd

output_df = pd.read_excel("bpd_time_output.xlsx")

# Initialize dictionaries to store the results
entropy_values = {}
acorr_values = {}


for column in output_df.columns:
    
    clean_series = output_df[column].dropna()
    
    # Compute Sample Entropy
    entropy = nolds.sampen(clean_series)

    # Compute Autocorrelation for lag 1
    acorr = np.corrcoef(np.array([clean_series[:-1], clean_series[1:]]))[0, 1]

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