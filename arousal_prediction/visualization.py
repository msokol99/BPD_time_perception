import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import logging
from transformers import AutoTokenizer
import torch_directml
from RobertaArousalClass import RobertaArousalClass


############# SPECIFICATIONS ###############
logging.basicConfig(level=logging.ERROR)
torch.manual_seed(42)

############### SETTINGS ####################
base_model_dir = "sdadas/polish-roberta-base-v2"
trained_model_dir = 'arousal_prediction/output_model/arousal_model.pth'

BATCH_SIZE = 16
MAX_LEN = 256
tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
tokenizer.pad_token = "<pad>"

model_p = RobertaArousalClass()
model_p.load_state_dict(torch.load(trained_model_dir))
model_p.eval()

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
