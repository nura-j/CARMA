import json
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from src.dataset import TokenSpan


class SavedPredictionsDataset(Dataset):
    """Dataset for loading saved predictions."""

    def __init__(self, predictions):
        """
        Initialize the dataset.
        Args:
            predictions (list): List of dictionaries containing the data.
                                Each dictionary must have `input_ids`, `attention_mask`,
                                `labels`, and `annotations`.
        """
        self.predictions = predictions

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        item = self.predictions[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["label"]),
            "annotations": [
                TokenSpan(**span) for span in item["annotations"]['token_spans']
            ],  # Convert dictionary back to TokenSpan if needed
            'reduction_percentage': item["annotations"]['reduction_percentage']
        }


def load_saved_predictions(file_path: str, batch_size: int = 16) -> DataLoader:
    """
    Load saved predictions from a JSON file and return a DataLoader.
    Args:
        file_path (str): Path to the saved predictions JSON file.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader instance for the saved predictions.
    """
    # Read JSON file
    with open(file_path, "r") as f:
        predictions = json.load(f)

    # Create the dataset
    dataset = SavedPredictionsDataset(predictions)

    # Define a collate function for padding
    def collate_fn(batch):
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
        attention_masks = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = torch.stack([item["labels"] for item in batch])
        annotations = [item["annotations"] for item in batch]  # Annotations are not padded
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "annotations": annotations,
            "reduction_percentage": [item['reduction_percentage'] for item in batch]
        }

    # Return the DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)