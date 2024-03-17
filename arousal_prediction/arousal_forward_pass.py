import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from transformers import AutoTokenizer
import torch_directml
from AutobiographicalDataClass import AutobiographicalData
from RobertaArousalClass import RobertaArousalClass


############# SPECIFICATIONS ###############
logging.basicConfig(level=logging.ERROR)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
device = torch_directml.device()

############### SETTINGS ####################
output_base_dir = 'arousal_prediction/output_corpora'
input_base_dir = 'input_corpora/borderline'

base_model_dir = "sdadas/polish-roberta-base-v2"
trained_model_dir = 'arousal_prediction/output_model/arousal_model.pth'

BATCH_SIZE = 16
MAX_LEN = 256
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
tokenizer.pad_token = "<pad>"

model_p = RobertaArousalClass()
model_p.load_state_dict(torch.load(trained_model_dir, map_location=device))
model_p.to(device)
model_p.eval()

# Process each file in the input directory
for subdir, dirs, files in os.walk(input_base_dir):
    for file in files:
        # Skip if not an Excel file
        if not file.endswith('.xlsx') and not file.endswith('.xls'):
            continue
        
        input_file_path = os.path.join(subdir, file)
        df = pd.read_excel(input_file_path)

        dataset = AutobiographicalData(dataframe=df, tokenizer=tokenizer, max_len=MAX_LEN)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        inputs = np.array(df.sentence)

        predictions = []

        for batch in tqdm(data_loader, total=len(data_loader), desc=f'Predicting on {file}'):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            with torch.no_grad():
                outputs = model_p(ids, attention_mask=mask, token_type_ids=token_type_ids)

            _, predicted_class = torch.max(outputs, 1)
            predictions.extend(predicted_class.cpu().tolist())

        all_predictions_np = np.array(predictions)
        output_df = pd.DataFrame({'sentence': inputs, 'arousal': all_predictions_np})

        # Construct output directory mirroring the input structure
        output_subdir = subdir.replace(input_base_dir, output_base_dir)
        os.makedirs(output_subdir, exist_ok=True)
        output_file = os.path.join(output_base_dir, os.path.splitext(file)[0] + '_arousal.csv')
        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Predictions written to {output_file}")
