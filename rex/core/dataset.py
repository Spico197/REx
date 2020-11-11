from torch.utils import data as td


class MultiClassDataset(td.Dataset):
    """Dataset for multi-class classification"""
    def __init__(self):
        raise NotImplementedError


class MultiClassMultiLabelDataset(td.Dataset):
    """Dataset for multi-class multi-label classification"""
    def __init__(self):
        raise NotImplementedError
