from pathlib import Path
from typing import Callable, Optional, Union

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from rex.utils.deprecation import deprecation_warning
from rex.utils.io import dump_pickle, load_pickle
from rex.utils.logging import logger


class DataManager(object):
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
        dump_cache_dir: Optional[str] = None,
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
        self.dump_cache_dir = dump_cache_dir
        if self.dump_cache_dir is not None:
            self.dump_cache_dir = Path(self.dump_cache_dir)
            if not self.dump_cache_dir.exists():
                self.dump_cache_dir.mkdir()
            else:
                logger.warning(
                    f"Cached dir exists: {str(self.dump_cache_dir.absolute())}"
                )

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
        result_name = DataManager._NAME_TO_NORMALIZED.get(dataset_name)
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
            if self.dump_cache_dir:
                cache_filepath = self.dump_cache_dir.joinpath(f"{dataset_name}.cache")
            else:
                cache_filepath = None

            if (
                not self.debug_mode
                and cache_filepath is not None
                and cache_filepath.exists()
            ):
                dataset = load_pickle(cache_filepath)
                logger.info(
                    f"Load cached {dataset_name} dataset from {str(cache_filepath)}"
                )
            else:
                if self.use_stream_transform:
                    dataset = self.dataset_class(
                        self.load_fn(filepath), self.transform, debug=self.debug_mode
                    )
                else:
                    dataset = self.dataset_class(
                        self.transform(self.load_fn(filepath), debug=self.debug_mode)
                    )
                if self.dump_cache_dir and not cache_filepath.exists():
                    logger.info(
                        f"Dump cached {dataset_name} dataset tp {str(cache_filepath)}"
                    )
                    dump_pickle(dataset, cache_filepath)
            self._update_dataset(dataset_name, dataset)
        return dataset

    @staticmethod
    def guess_eval_from_name(dataset_name: str):
        _dataset_name = DataManager._get_normalized_dataset_name(dataset_name)
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
            loader = self.prepare_loader(dataset, is_eval=is_eval, epoch_idx=epoch)
            self._update_loader(dataset_name, loader)
        return loader

    def prepare_loader(self, dataset, is_eval=True, epoch_idx=0, **kwargs):
        shuffle_flag = self.eval_shuffle if is_eval else self.train_shuffle
        batch_size = self.eval_batch_size if is_eval else self.train_batch_size

        if self.distributed_mode:
            sampler = DistributedSampler(dataset, shuffle=shuffle_flag)
            sampler.set_epoch(epoch_idx)
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
            **kwargs,
        )
        return loader

    def load(self, dataset_name: str):
        dataset = self.load_dataset(dataset_name)
        loader = self.load_loader(dataset_name)
        return dataset, loader

    def __getattr__(self, name: str):
        """Get dataset or loader in a lazy way

        Args:
            name (~str): dataset_name + "_set" or "_loader":
                [
                    train_set, train_loader,
                    train_eval_set, train_eval_loader,
                    dev_set, dev_loader,
                    test_set, test_loader
                ]

        Returns:
            dataset or dataloader

        Raises:
            AttributeError if dataset or data loader does not exist
        """
        if name.endswith("_set"):
            return self.load_dataset(name[:-4])
        elif name.endswith("_loader"):
            return self.load_loader(name[:-7])
        else:
            raise AttributeError(f"Attribute {name} does not exist")


class CachedManager(DataManager):
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

        deprecation_warning(self.__class__.__name__, DataManager.__name__)


class StreamTransformManager(DataManager):
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

        deprecation_warning(self.__class__.__name__, DataManager.__name__)
