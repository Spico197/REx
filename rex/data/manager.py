from typing import Callable, Optional, Union

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from rex.utils.logging import logger


class Manager(object):
    DATASET_NAMES = {
        "train": ["train", "training"],
        "dev": ["dev", "development", "val", "validation", "validating", "validate"],
        "test": ["test", "testing"],
        "train_eval": ["train_eval", "train4eval", "eval_train"],
    }
    _NAME_TO_NORMALIZED = {}
    for normalized_name, names in DATASET_NAMES.items():
        for name in names:
            _NAME_TO_NORMALIZED[name] = normalized_name

    def __init__(
        self,
        train_filepath: str,
        dev_filepath: str,
        test_filepath: str,
        dataset_class: Dataset,
        transform: Callable,
        load_fn: Callable,
        train_batch_size: int,
        eval_batch_size: int,
        collate_fn: Callable,
        use_stream_transform: bool,
        train_shuffle: Optional[bool] = True,
        eval_shuffle: Optional[bool] = False,
        num_workers: Optional[int] = 0,
        debug_mode: Optional[bool] = False,
        distributed_mode: Optional[bool] = False,
        # lazy loading when using
        load_train_data: Optional[bool] = False,
        load_dev_data: Optional[bool] = False,
        load_test_data: Optional[bool] = False,
        load_train_eval_data: Optional[bool] = False,
    ):
        self._dataset_name_to_filepath = {
            "train": train_filepath,
            "dev": dev_filepath,
            "test": test_filepath,
            "train_eval": train_filepath,
        }
        self._dataset_name_to_dataset = {
            "train": None,
            "dev": None,
            "test": None,
            "train_eval": None,
        }
        self._dataset_name_to_loader = {
            "train": None,
            "dev": None,
            "test": None,
            "train_eval": None,
        }

        self.dataset_class = dataset_class
        self.transform = transform
        self.load_fn = load_fn
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.collate_fn = collate_fn
        self.use_stream_transform = use_stream_transform
        self.train_shuffle = train_shuffle
        self.eval_shuffle = eval_shuffle
        self.num_workers = num_workers
        self.debug_mode = debug_mode
        self.distributed_mode = distributed_mode

        if load_train_data:
            self.load("train")
        if load_dev_data:
            self.load("dev")
        if load_test_data:
            self.load("test")
        if load_train_eval_data:
            self.load("train_eval")

    @staticmethod
    def _get_normalized_dataset_name(dataset_name: str) -> str:
        dataset_name = dataset_name.lower().strip()
        result_name = Manager._NAME_TO_NORMALIZED.get(dataset_name)
        if result_name is None:
            raise ValueError(f"Dataset: {dataset_name} cannot be normalized!")
        return result_name

    def _get_dataset_from_name(self, dataset_name: str):
        _dataset_name = self._get_normalized_dataset_name(dataset_name)
        return self._dataset_name_to_dataset[_dataset_name]

    def _get_loader_from_name(self, dataset_name: str):
        _dataset_name = self._get_normalized_dataset_name(dataset_name)
        return self._dataset_name_to_loader[_dataset_name]

    def _get_filepath_from_name(self, dataset_name: str):
        _dataset_name = self._get_normalized_dataset_name(dataset_name)
        return self._dataset_name_to_filepath[_dataset_name]

    def _update_dataset(self, dataset_name: str, dataset: Dataset):
        _dataset_name = self._get_normalized_dataset_name(dataset_name)
        self._dataset_name_to_dataset[_dataset_name] = dataset

    def _update_loader(self, dataset_name: str, loader: DataLoader):
        _dataset_name = self._get_normalized_dataset_name(dataset_name)
        self._dataset_name_to_loader[_dataset_name] = loader

    def load_dataset(self, dataset_name: str):
        dataset = self._get_dataset_from_name(dataset_name)
        if dataset is None:
            filepath = self._get_filepath_from_name(dataset_name)
            if self.use_stream_transform:
                dataset = self.dataset_class(
                    self.load_fn(filepath), self.transform, debug=self.debug_mode
                )
            else:
                dataset = self.data_class(
                    self.transform(self.load_fn(filepath), debug=self.debug_mode)
                )
            self._update_dataset(dataset_name, dataset)
        return dataset

    @staticmethod
    def guess_eval_from_name(dataset_name: str):
        _dataset_name = Manager._get_normalized_dataset_name(dataset_name)
        return _dataset_name == "train"

    def load_loader(
        self,
        dataset_name: str,
        is_eval: Optional[Union[str, bool]] = "guessing",
        epoch: Optional[int] = 0,
    ):
        loader = self._get_loader_from_name(dataset_name)
        if loader is None:
            if isinstance(is_eval, str) and is_eval == "guessing":
                is_eval = self.guess_eval_from_name(dataset_name)
            elif not isinstance(is_eval, bool):
                raise ValueError(f"Not recognized `is_eval`: {is_eval}")

            dataset = self.load_dataset(dataset_name)
            shuffle_flag = self.eval_shuffle if is_eval else self.train_shuffle
            batch_size = self.eval_batch_size if is_eval else self.train_batch_size

            if self.distributed_mode:
                sampler = DistributedSampler(dataset, shuffle=shuffle_flag)
                sampler.set_epoch(epoch)
            elif shuffle_flag:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
            self._update_loader(dataset_name, loader)
        return loader

    def load(self, dataset_name: str):
        dataset = self.load_dataset(dataset_name)
        loader = self.load_loader(dataset_name)
        return dataset, loader


class CachedManager(Manager):
    def __init__(
        self,
        train_filepath: str,
        dev_filepath: str,
        test_filepath: str,
        dataset_class: Dataset,
        transform: Callable,
        load_fn: Callable,
        train_batch_size: int,
        eval_batch_size: int,
        collate_fn: Callable,
        train_shuffle: Optional[bool] = True,
        eval_shuffle: Optional[bool] = False,
        num_workers: Optional[int] = 0,
        debug_mode: Optional[bool] = False,
        distributed_mode: Optional[bool] = False,
        load_train_data: Optional[bool] = False,
        load_dev_data: Optional[bool] = False,
        load_test_data: Optional[bool] = False,
        load_train_eval_data: Optional[bool] = False,
    ):
        super().__init__(
            train_filepath,
            dev_filepath,
            test_filepath,
            dataset_class,
            transform,
            load_fn,
            train_batch_size,
            eval_batch_size,
            collate_fn,
            use_stream_transform=False,
            train_shuffle=train_shuffle,
            eval_shuffle=eval_shuffle,
            num_workers=num_workers,
            debug_mode=debug_mode,
            distributed_mode=distributed_mode,
            load_train_data=load_train_data,
            load_dev_data=load_dev_data,
            load_test_data=load_test_data,
            load_train_eval_data=load_train_eval_data,
        )

        logger.warning(
            "CachedManager is deprecated and will be removed from the stablized version, please change to `Manager` instead."
        )


class StreamTransformManager(Manager):
    def __init__(
        self,
        train_filepath: str,
        dev_filepath: str,
        test_filepath: str,
        dataset_class: Dataset,
        transform: Callable,
        load_fn: Callable,
        train_batch_size: int,
        eval_batch_size: int,
        collate_fn: Callable,
        train_shuffle: Optional[bool] = True,
        eval_shuffle: Optional[bool] = False,
        num_workers: Optional[int] = 0,
        debug_mode: Optional[bool] = False,
        distributed_mode: Optional[bool] = False,
        load_train_data: Optional[bool] = False,
        load_dev_data: Optional[bool] = False,
        load_test_data: Optional[bool] = False,
        load_train_eval_data: Optional[bool] = False,
    ):
        super().__init__(
            train_filepath,
            dev_filepath,
            test_filepath,
            dataset_class,
            transform,
            load_fn,
            train_batch_size,
            eval_batch_size,
            collate_fn,
            use_stream_transform=True,
            train_shuffle=train_shuffle,
            eval_shuffle=eval_shuffle,
            num_workers=num_workers,
            debug_mode=debug_mode,
            distributed_mode=distributed_mode,
            load_train_data=load_train_data,
            load_dev_data=load_dev_data,
            load_test_data=load_test_data,
            load_train_eval_data=load_train_eval_data,
        )

        logger.warning(
            "StreamTransformManager is deprecated and will be removed from the stablized version, please change to `Manager` instead."
        )
