import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class RobertaArousalClass(nn.Module):
    """
    A custom neural network class that integrates a pre-trained Polish RoBERTa model for sentiment analysis.
    
    Attributes:
        model_config: The configuration settings of the RoBERTa model.
        roberta: The loaded RoBERTa model with its pre-trained weights.
        dropout1: A dropout layer to reduce overfitting in the first set of layers.
        linear1: A linear layer to transform the input dimension.
        relu1: A ReLU activation function following the first linear transformation.
        bn1: Batch normalization for the first transformed output.
        dropout2: Another dropout layer to reduce overfitting in the subsequent set of layers.
        linear2: A second linear layer to further transform the data.
        relu2: A ReLU activation function following the second linear transformation.
        bn2: Batch normalization for the second transformed output.
    """
    
    def __init__(self):
        """
        Initializes the RobertaArousalClass with a pre-defined architecture and model configuration.
        """
        super(RobertaArousalClass, self).__init__()

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
