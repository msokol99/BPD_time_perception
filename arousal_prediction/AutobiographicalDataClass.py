import torch
from torch.utils.data import Dataset

class AutobiographicalData(Dataset):
    """
    A custom PyTorch Dataset for loading autobiographical sentences.

    Attributes:
        tokenizer: The tokenizer used to process the text data.
        data: The dataframe containing the dataset.
        text: The specific column in the dataframe that contains the text.
        max_len: The maximum length for the tokenized text.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the AutobiographicalData dataset with the dataframe, tokenizer, and maximum token length.

        Args:
            dataframe: A Pandas dataframe containing the dataset.
            tokenizer: A tokenizer instance to convert text into tokens.
            max_len: An integer specifying the maximum length of the tokenized sequences.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['sentence']  # Assuming 'sentence' is the column with text data
        self.max_len = max_len

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            An integer count of the number of items in the dataset.
        """
        return len(self.text)

    def __getitem__(self, index):
        """
        Retrieves an item by its index.

        Args:
            index: An integer index corresponding to the item in the dataset.

        Returns:
            A dictionary containing tokenized input IDs, attention masks, and token type IDs.
        """
        text = str(self.text[index])
        # Preprocess the text by converting to lowercase and splitting into words
        text = " ".join(word.lower() for word in text.split())

        # Tokenize the text and obtain necessary inputs for the model
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        # Extract and convert inputs to tensors
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }
