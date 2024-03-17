import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from SentimentDataClass import SentimentData
from RobertaValenceClass import RobertaValenceClass
import torch_directml


############# SPECIFICATIONS ###############
logging.basicConfig(level=logging.ERROR)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
# Setting up the device for GPU usage
device = torch_directml.device()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


############### SETTINGS ####################
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-05
EPSILON = 1e-8
WEIGHT_DECAY = 0.01

model_dir = "sdadas/polish-roberta-base-v2"
save_dir = "valence_prediction/output_model/valence_model.pth"

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = "<pad>"


############### DATASET LOADING ###############
train_df = pd.read_csv('valence_prediction/data/train_data.csv')
val_df = pd.read_csv('valence_prediction/data/val_data.csv')

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': True,
                }

val_params = {'batch_size': int(BATCH_SIZE/2),
                'shuffle': True,
                'num_workers': 0,
                'pin_memory': True,
                }

training_loader = DataLoader(SentimentData(train_df, tokenizer, MAX_LEN), **train_params)
validation_loader = DataLoader(SentimentData(val_df, tokenizer, MAX_LEN), **val_params)

model = RobertaValenceClass()
model.to(device)


############### OPTIMIZER AND SCHEDULER SETTINGS ########################
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON, weight_decay=WEIGHT_DECAY)

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int((len(training_loader) * EPOCHS)*0.05),
                                            num_training_steps=len(training_loader) * EPOCHS)


######################### TRAINING ######################################
def train(optimizer, scheduler, epochs, training_loader, validation_loader):
    best_accuracy = 0

    model.train()

    for epoch_num in range(epochs):
        total_loss_train = 0
        correct_predictions_train = 0
        total_samples_train = 0

        for _, data in tqdm(enumerate(training_loader, 0), leave=True, total=len(training_loader), desc=f'Training in epoch {epoch_num + 1}'):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            targets = data['targets'].to(device, dtype=torch.long)  # Change to long for classification

            outputs = model(ids, mask, token_type_ids)

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)

            total_loss_train += loss.item()

            _, predicted_train = torch.max(outputs, 1)
            correct_predictions_train += (predicted_train == targets).sum().item()
            total_samples_train += targets.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy_train = correct_predictions_train / total_samples_train

        total_loss_val = 0
        correct_predictions_val = 0
        total_samples_val = 0

        model.eval()

        with torch.no_grad():
            for _, data in tqdm(enumerate(validation_loader, 0), total=len(validation_loader), leave=True,
                                desc=f'Validation in epoch {epoch_num + 1}'):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

                targets_val = data['targets'].to(device, dtype=torch.long)
                outputs_val = model(ids, mask, token_type_ids)

                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs_val, targets_val)

                total_loss_val += loss.item()

                _, predicted_val = torch.max(outputs_val, 1)
                correct_predictions_val += (predicted_val == targets_val).sum().item()
                total_samples_val += targets_val.size(0)

        accuracy_val = correct_predictions_val / total_samples_val

        if best_accuracy < accuracy_val:
            best_accuracy = accuracy_val
            torch.save(model.state_dict(), save_dir)
            print(f'Saved model for epoch {epoch_num + 1}')

        print(
            f'Summary of epoch {epoch_num + 1} | Train Loss: {total_loss_train / len(training_loader): .10f} '
            f'| Train Accuracy: {accuracy_train: .4f} | Validation Loss: {total_loss_val / len(validation_loader): .10f} | Validation Accuracy: {accuracy_val: .4f}')

        scheduler.step()


train(optimizer, scheduler, EPOCHS, training_loader, validation_loader)