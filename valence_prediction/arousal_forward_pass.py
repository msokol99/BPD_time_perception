import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import PreTrainedTokenizerFast
import torch_directml
from AutobiographicalDataClass import AutobiographicalData
from RobertaArousalClass import RobertaArousalClass


############# SPECIFICATIONS ###############
# DirectML GPU acceleration for AMD GPU only; for NVIDIA use CUDA
logging.basicConfig(level=logging.ERROR)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
device = torch_directml.device()


############### SETTINGS ####################
BATCH_SIZE = 16
MAX_LEN = 256
tokenizer = PreTrainedTokenizerFast(tokenizer_file="roberta_base_transformers/tokenizer.json")
tokenizer.pad_token = "<pad>"

trained_model_dir = 'arousal_prediction/output_model/model_Marzec_3.pth'
autobiographical_data = 'Corpora/corpus_for_cleanup.xlsx'


##############################################
model_p=RobertaArousalClass()
model_p.load_state_dict(torch.load(trained_model_dir))
model_p.eval()

df = pd.read_excel(autobiographical_data)
# Instantiate the SentimentData class
dataset = AutobiographicalData(dataframe=df, tokenizer=tokenizer, max_len=MAX_LEN)
# Create a DataLoader for batching
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
# Take an array of sentences from the df as an input
inputs = np.array(df.sentence)

# Initialize a list to store predictions
predictions = []

# Iterate through batches
for batch in tqdm(data_loader, total=len(data_loader), desc='Predicting on new data'):
    # Move inputs to the device used for training (e.g., GPU)
    ids = batch['ids']
    mask = batch['mask']
    token_type_ids = batch['token_type_ids']

    # Forward pass
    with torch.no_grad():
        outputs = model_p(ids, attention_mask=mask, token_type_ids=token_type_ids)

    # Get the predicted class indices
    _, predicted_class = torch.max(outputs, 1)

    # Append predictions to the list
    predictions.extend(predicted_class.tolist())

# Convert the list to a NumPy array if needed
all_predictions_np = np.array(predictions)

# Create a DataFrame with sentences and predictions as columns
result_df = pd.DataFrame({'sentence': inputs, 'arousal': all_predictions_np})

#result_df.to_csv('C:/Users/marta/OneDrive/Desktop/arousal_prediction/data/output_file.csv', index=False, encoding='utf-8-sig')
result_df.to_csv('C:/Users/marta/OneDrive/Desktop/Dane/Autobiografie/Literature/arousal_output_file.csv', index=False, encoding='utf-8-sig')

output_csv=pd.read_csv('C:/Users/marta/OneDrive/Desktop/Dane/Autobiografie/Literature/arousal_output_file.csv')

#output_csv=output_csv.sort_values(by='arousal', ascending=False)

#output_csv = output_csv.drop(columns='arousal')

output_csv.to_excel('C:/Users/marta/OneDrive/Desktop/Dane/Autobiografie/Literature/arousal_output_file.xlsx', index=False)

from torch.utils.tensorboard import SummaryWriter

# Create a dummy input tensor to visualize the model architecture
dummy_input = {
    'ids': torch.zeros((1, 5), dtype=torch.long),  # Adjust the sequence length (5) as needed
    'mask': torch.ones((1, 5), dtype=torch.long),
    'token_type_ids': torch.zeros((1, 5), dtype=torch.long)
}

# Enable TensorBoard
writer = SummaryWriter()

# Visualize the model architecture
writer.add_graph(model_p, (dummy_input['ids'], dummy_input['mask'], dummy_input['token_type_ids']))

# Close the SummaryWriter
writer.close()

