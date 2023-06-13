from abc import ABC, abstractmethod
from typing import Any, List, Optional

from rex.utils.logging import logger
from rex.utils.progress_bar import pbar


class TransformBase(ABC):
    @abstractmethod
    def transform(self, lines: List, *args, **kwargs):
        raise NotImplementedError

    def predict_transform(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any):
        return self.transform(*args, **kwargs)


class CachedTransformBase(TransformBase):
    @abstractmethod
    def transform(self, loader: List, **kwargs) -> List[dict]:
        raise NotImplementedError

    def predict_transform(self, *args, **kwargs) -> List[dict]:
        raise NotImplementedError

    def __call__(
        self,
        dataset: List,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        debug_len: Optional[int] = 50,
        disable_pbar: Optional[bool] = False,
        num_samples: Optional[int] = 3,
        **kwargs,
    ) -> List[dict]:
        if debug:
            dataset = dataset[:debug_len]
        transform_loader = pbar(dataset, desc=desc, disable=disable_pbar)
        final_data: list = self.transform(transform_loader, **kwargs)
        logger.info(transform_loader)
        logger.debug(f"#Data: {len(final_data)}")
        # set `colors=False` temporarily in case of any
        #   unexpected color tags (e.g. <tag>, </tag>) are included in the data
        logger.opt(colors=False).debug(f"{final_data[:num_samples]}")
        return final_data


class CachedTransformOneBase(TransformBase):
    def __init__(self, ignore_error: bool = True) -> None:
        super().__init__()

        self.ignore_error = ignore_error

    @abstractmethod
    def transform(self, instance: dict, **kwargs) -> dict:
        raise NotImplementedError

    def predict_transform(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def __call__(
        self,
        dataset: List,
        desc: Optional[str] = "Transform",
        debug: Optional[bool] = False,
        debug_len: Optional[int] = 50,
        disable_pbar: Optional[bool] = False,
        num_samples: Optional[int] = 3,
        **kwargs,
    ) -> List[dict]:
        if debug:
            dataset = dataset[:debug_len]
        transform_loader = pbar(dataset, desc=desc, disable=disable_pbar)

        final_data = []
        num_err = 0
        for ins in transform_loader:
            try:
                transformed_one = self.transform(ins, **kwargs)
                if transformed_one is None:
                    continue
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception:
                num_err += 1
                logger.error(f"#{num_err} Error when transforming {ins}")
                if self.ignore_error:
                    continue
                else:
                    raise

            if isinstance(transformed_one, dict):
                final_data.append(transformed_one)
            elif isinstance(transformed_one, list):
                final_data.extend(transformed_one)
            else:
                raise RuntimeError(f"{type(transformed_one)} unsupported!")

        logger.info(transform_loader)
        logger.debug(f"#Data: {len(final_data)}")
        # set `colors=False` temporarily in case of any
        #   unexpected color tags (e.g. <tag>, </tag>) are included in the data
        logger.opt(colors=False).debug(f"{final_data[:num_samples]}")
        return final_data


class StreamTransformOneBase(TransformBase):
    @abstractmethod
    def transform(self, instance: dict, **kwargs) -> dict:
        raise NotImplementedError

    def predict_transform(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def __call__(
        self, instance: dict, dataset_name: str = None, **kwargs
    ) -> List[dict]:
        return self.transform(instance, dataset_name=dataset_name, **kwargs)
