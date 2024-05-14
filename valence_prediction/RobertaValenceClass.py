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

        # Initialize the model configuration using a pre-trained RoBERTa model.
        self.model_config = RobertaConfig.from_pretrained('sdadas/polish-roberta-base-v2')

        # Load the pre-trained RoBERTa model with the initialized configuration.
        self.roberta = RobertaModel.from_pretrained('sdadas/polish-roberta-base-v2', config=self.model_config)

        # Define network layers and dropout rates for regularization
        self.dropout1 = nn.Dropout(0.5)  # First dropout layer with a 50% drop rate
        self.linear1 = nn.Linear(768, 512)  # Linear layer reducing dimensionality to 512
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization for the output of the first linear layer

        self.dropout2 = nn.Dropout(0.3)  # Second dropout layer with a 30% drop rate
        self.linear2 = nn.Linear(512, 3)  # Linear layer reducing dimensionality to 3
        self.relu2 = nn.ReLU()  # ReLU activation function for the second linear layer
        self.bn2 = nn.BatchNorm1d(3)  # Batch normalization for the output of the second linear layer

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Defines the forward pass of the model.

        Args:
            input_ids: Tensor of input IDs.
            attention_mask: Tensor representing attention masks to avoid focusing on padding.
            token_type_ids: Tensor of segment IDs to distinguish different types of tokens.

        Returns:
            The output of the neural network after processing the input through RoBERTa and additional layers.
        """
        # Obtain the outputs from the RoBERTa model.
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = outputs.pooler_output  # Extract the pooled output features.

        # Sequentially pass the output through the defined layers with activations and normalizations
        out = self.dropout1(output)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.bn1(out)

        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        return out

