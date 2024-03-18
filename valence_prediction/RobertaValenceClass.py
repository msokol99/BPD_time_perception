import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class RobertaValenceClass(nn.Module):
    """
    A neural network class that utilizes a pre-trained RoBERTa model for valence prediction.

    This class is designed to fine-tune the RoBERTa model for a specific task related to valence
    prediction, which is a common task in sentiment analysis or affective computing.

    Attributes:
        model_config (RobertaConfig): Configuration settings for the RoBERTa model.
        roberta (RobertaModel): The pre-trained RoBERTa model.
        dropout (nn.Dropout): Dropout layer to reduce overfitting.
        linear (nn.Linear): Linear layer to map hidden states to output classes.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(self):
        """
        Initializes the RobertaValenceClass with a pre-defined architecture and model configuration.
        """
        super(RobertaValenceClass, self).__init__()

        # Load the model configuration specific to 'sdadas/polish-roberta-base-v2'
        self.model_config = RobertaConfig.from_pretrained('sdadas/polish-roberta-base-v2')

        # Load the pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained('sdadas/polish-roberta-base-v2', config=self.model_config)

        # Define a dropout layer to prevent overfitting with a dropout rate of 0.3
        self.dropout = nn.Dropout(0.3)
        # Linear layer to map the pooled output to three classes (adjust according to your task)
        self.linear = nn.Linear(768, 3)
        # ReLU activation function to introduce non-linearity
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input IDs for the RoBERTa model.
            attention_mask (torch.Tensor): Attention mask to avoid attention on padding.
            token_type_ids (torch.Tensor): Token type IDs to distinguish different sequences.

        Returns:
            torch.Tensor: The output logits from the final linear layer.
        """
        # Get the outputs from the RoBERTa model
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Use the pooled output which is a summary of the content, as per RoBERTa's design
        output = outputs.pooler_output

        # Pass the output through the dropout and linear layers, and then apply the ReLU activation
        out = self.dropout(output)
        out = self.linear(out)
        out = self.relu(out)

        return out

