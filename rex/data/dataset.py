from typing import Iterable, Iterator, Optional

from torch.utils.data import Dataset, IterableDataset


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


class StreamTransformDataset(Dataset):
    def __init__(
        self, data: Iterable, transform, debug: Optional[bool] = False
    ) -> None:
        super().__init__()
        if debug:
            data = data[:128]
        self.data = data
        self.transform = transform

    def __getitem__(self, index: int):
        return self.transform(self.data[index])

    def __len__(self) -> int:
        return len(self.data)


class StreamReadDataset(IterableDataset):
    def __init__(
        self, data_iterator: Iterator, transform, debug: Optional[bool] = False
    ) -> None:
        super().__init__()

        self.transform = transform
        self.data_iterator = data_iterator
        self.debug = debug
        self.cnt = 0

    def __iter__(self):
        for item in self.data_iterator:
            if self.debug and self.cnt >= 500:
                raise StopIteration
            yield self.transform(item)
            self.cnt += 1
