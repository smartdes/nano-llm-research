import torch
from typing import List, Tuple

class TextDataset:
    """
    Character-level text dataset for training autoregressive language models.
    Handles text loading, vocabulary creation, character tokenization, and batch sampling.
    """
    def __init__(self, filepath: str, block_size: int, train_split: float = 0.9):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.block_size = block_size

        # Encode entire text into integer tokens
        data_ints = [self.stoi[c] for c in self.text]
        self.data = torch.tensor(data_ints, dtype=torch.long)

        # Split into training and validation sets
        split_idx = int(train_split * len(self.data))
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    def encode(self, s: str) -> List[int]:
        """Convert a string into a list of token integers."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, indices: List[int]) -> str:
        """Convert a list of token integers back into text."""
        return ''.join([self.itos[i] for i in indices if i in self.itos])

    def get_batch(self, split: str = 'train', batch_size: int = 32, device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of inputs (x) and targets (y)."""
        data = self.train_data if split == 'train' else self.val_data
        max_idx = len(data) - self.block_size
        if max_idx <= 0:
            raise ValueError(
                f"Split '{split}' size ({len(data)}) is too small for block_size ({self.block_size})."
            )

        ix = torch.randint(max_idx, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])

        if device is not None:
            x, y = x.to(device), y.to(device)
        return x, y
