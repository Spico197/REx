from typing import Iterable

from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, data: Iterable) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
