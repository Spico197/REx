from typing import Callable, Optional

from torch.utils.data import DataLoader, Dataset


class CachedManager(object):
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
        num_workers: Optional[int] = 4,
        debug_mode: Optional[bool] = False,
        load_train_data: Optional[bool] = True,
        load_dev_data: Optional[bool] = True,
        load_test_data: Optional[bool] = True,
    ):
        self.collate_fn = collate_fn

        if load_train_data:
            self.train_set = dataset_class(
                transform(load_fn(train_filepath), debug=debug_mode)
            )
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=train_batch_size,
                shuffle=train_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            self.train_eval_loader = DataLoader(
                self.train_set,
                batch_size=eval_batch_size,
                shuffle=eval_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        if load_dev_data:
            self.dev_set = dataset_class(
                transform(load_fn(dev_filepath), debug=debug_mode)
            )
            self.dev_loader = DataLoader(
                self.dev_set,
                batch_size=eval_batch_size,
                shuffle=eval_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        if load_test_data:
            self.test_set = dataset_class(
                transform(load_fn(test_filepath), debug=debug_mode)
            )
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=eval_batch_size,
                shuffle=eval_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )


class StreamTransformManager(object):
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
        num_workers: Optional[int] = 4,
        debug_mode: Optional[bool] = False,
        load_train_data: Optional[bool] = True,
        load_dev_data: Optional[bool] = True,
        load_test_data: Optional[bool] = True,
    ):
        self.collate_fn = collate_fn

        if load_train_data:
            self.train_set = dataset_class(
                load_fn(train_filepath), transform, debug=debug_mode
            )
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=train_batch_size,
                shuffle=train_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            self.train_eval_loader = DataLoader(
                self.train_set,
                batch_size=eval_batch_size,
                shuffle=eval_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        if load_dev_data:
            self.dev_set = dataset_class(
                load_fn(dev_filepath), transform, debug=debug_mode
            )
            self.dev_loader = DataLoader(
                self.dev_set,
                batch_size=eval_batch_size,
                shuffle=eval_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        if load_test_data:
            self.test_set = dataset_class(
                load_fn(test_filepath), transform, debug=debug_mode
            )
            self.test_loader = DataLoader(
                self.test_set,
                batch_size=eval_batch_size,
                shuffle=eval_shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
