import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class RobertaValenceClass(nn.Module):
    def __init__(self):  # Adjust num_classes based on your task
        super(RobertaValenceClass, self).__init__()

        # Load model configuration
        self.model_config = RobertaConfig.from_pretrained('sdadas/polish-roberta-base-v2')

        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained('sdadas/polish-roberta-base-v2', config=self.model_config)

        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = outputs.pooler_output  # Using 'pooler_output' instead of 'last_hidden_state'

        # Dense Layers
        out = self.dropout(output)
        out = self.linear(out)
        out = self.relu(out)

        return out
