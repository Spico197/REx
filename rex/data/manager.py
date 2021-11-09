from torch.utils.data import DataLoader


class CachedManager(object):
    def __init__(
        self,
        train_filepath,
        dev_filepath,
        test_filepath,
        dataset_class,
        transform,
        load_fn,
        train_batch_size,
        eval_batch_size,
        collate_fn,
        train_shuffle=True,
        eval_shuffle=False,
        num_workers=4,
        debug_mode=False,
    ):
        self.collate_fn = collate_fn

        self.train_set = dataset_class(
            transform(load_fn(train_filepath), debug=debug_mode)
        )
        self.dev_set = dataset_class(transform(load_fn(dev_filepath), debug=debug_mode))
        self.test_set = dataset_class(
            transform(load_fn(test_filepath), debug=debug_mode)
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        self.dev_loader = DataLoader(
            self.dev_set,
            batch_size=eval_batch_size,
            shuffle=eval_shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
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
        train_filepath,
        dev_filepath,
        test_filepath,
        dataset_class,
        transform,
        load_fn,
        train_batch_size,
        eval_batch_size,
        collate_fn,
        train_shuffle=True,
        eval_shuffle=False,
        num_workers=4,
        debug_mode=False,
    ):
        self.collate_fn = collate_fn

        self.train_set = dataset_class(
            load_fn(train_filepath), transform, debug=debug_mode
        )
        self.dev_set = dataset_class(load_fn(dev_filepath), transform, debug=debug_mode)
        self.test_set = dataset_class(
            load_fn(test_filepath), transform, debug=debug_mode
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
        self.dev_loader = DataLoader(
            self.dev_set,
            batch_size=eval_batch_size,
            shuffle=eval_shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=eval_batch_size,
            shuffle=eval_shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
