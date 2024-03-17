import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class RobertaArousalClass(nn.Module):
    def __init__(self):
        super(RobertaArousalClass, self).__init__()

        # Load model configuration
        self.model_config = RobertaConfig.from_pretrained('sdadas/polish-roberta-base-v2')

        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained('sdadas/polish-roberta-base-v2', config=self.model_config)

        self.dropout1 = nn.Dropout(0.5)  # Increased dropout rate
        self.linear1 = nn.Linear(768, 512)  # Decreased output size
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(512)  # Added batch normalization

        self.dropout2 = nn.Dropout(0.3)  # Increased dropout rate
        self.linear2 = nn.Linear(512, 3)  # Decreased output size
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(3)  # Added batch normalization

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = outputs.pooler_output  # Using 'pooler_output' instead of 'last_hidden_state'

        # Dense Layers
        out = self.dropout1(output)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.bn1(out)

        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        return out
