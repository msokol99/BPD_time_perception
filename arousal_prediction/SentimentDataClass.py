import torch
from torch.utils.data import Dataset

class SentimentData(Dataset):
    """
    A dataset class for sentiment analysis of Polish sentences.

    This class is designed to preprocess and tokenize the input text data for sentiment analysis tasks,
    particularly focusing on handling the sentiment associated with the arousal dimension in the input data.

    Parameters:
    - dataframe (pandas.DataFrame): The data frame containing the text data and their corresponding sentiment labels.
      The dataframe is expected to have at two columns: 'sentence' for the text and 'arousal' for the sentiment labels.
    - tokenizer: The tokenizer instance used for tokenizing the input texts.
    - max_len (int): The maximum length of the tokenized input sequences. Sequences longer than this length will be truncated.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the SentimentData object.

        Parameters:
        - dataframe (pandas.DataFrame): The dataset containing sentences and their corresponding arousal labels.
        - tokenizer: Tokenizer function for tokenizing text.
        - max_len (int): Maximum sequence length for padding/truncating the tokenized text sequences.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['sentence']
        self.targets = dataframe['arousal'].values.astype(int)
        self.max_len = max_len

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.text)

    def __getitem__(self, index):
        """
        Retrieves the tokenized text along with its attention mask, token type ids, and target label at a specified index.

        Parameters:
        - index (int): The index of the data point to be retrieved.

        Returns:
        - dict: A dictionary containing the following key-value pairs:
            - 'ids': Tensor of token ids to be fed to the model.
            - 'mask': Tensor representing attention mask values.
            - 'token_type_ids': Tensor of token type ids (segment ids).
            - 'targets': The label (target) of the sentiment class as a tensor.
        """
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.int)
        }
