import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs) -> dict:
        """Model forwarding
        To override this function, you must make sure `**kwargs` is in your args

        Returns:
            return a dict of results, loss and preds must be included
            example: {"loss": loss, "preds": preds}
        """
        raise NotImplementedError
