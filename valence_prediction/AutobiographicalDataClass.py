import torch
from torch.utils.data import Dataset

class AutobiographicalData(Dataset):
    """
    A custom PyTorch Dataset for processing autobiographical text data.

    This dataset takes in sentences from a dataframe, tokenizes them using a specified tokenizer,
    and returns the tokenized input along with its attention mask and token type ids for use in a model.

    Attributes:
        tokenizer: The tokenizer used to process the text.
        data: The dataframe containing the text and possibly other information.
        text: The column from the dataframe that contains text sentences.
        max_len: The maximum length for the tokenized text.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the AutobiographicalData dataset with the given dataframe, tokenizer, and maximum token length.

        Args:
            dataframe: A Pandas dataframe containing at least one column 'sentence' with text data.
            tokenizer: A tokenizer instance compatible with the text data and intended model.
            max_len: The maximum length of the tokenized sequences.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['sentence']
        self.max_len = max_len

    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: The number of sentences in the dataset.
        """
        return len(self.text)

    def __getitem__(self, index):
        """
        Retrieves a single tokenized item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the tokenized 'ids', 'mask', and 'token_type_ids' for the sentence.
        """
        text = str(self.text[index])
        # Normalize the text by converting to lowercase and removing extra spaces
        text = " ".join(word.lower() for word in text.split())

        # Tokenize the text with truncation and padding to max length
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,  # Adds special tokens (e.g., [CLS], [SEP])
            max_length=self.max_len,
            padding='max_length',  # Ensures all sequences are of the same length
            return_token_type_ids=True,  # Returns token type ids (useful for models like BERT)
            truncation=True  # Truncates to max_len if the sentence is longer
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # Convert the inputs to tensors
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }
