from typing import Iterable

from loguru import logger
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, data: Iterable) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class CachedBagREDataset(Dataset):
    def __init__(self, data_with_scopes) -> None:
        super().__init__()
        data, scopes = data_with_scopes
        self.data = data
        self.scopes = scopes

    def __getitem__(self, index: int):
        results = []
        for idx in self.scopes[index]:
            results.append(self.data[idx])
        return results

    def __len__(self) -> int:
        return len(self.scopes)
