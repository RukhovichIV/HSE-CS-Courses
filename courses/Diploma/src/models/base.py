import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class BaseUTSFModel:
    def __init__(self, **kwargs) -> None:
        raise RuntimeError("Not implemented")

    def fit(self, data: np.array) -> None:
        raise RuntimeError("Not implemented")

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        raise RuntimeError("Not implemented")


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleNoScaleScaler:
    def __init__(self, data: np.array):
        pass

    def transform(self, data: np.array):
        return data

    def inverse(self, data: np.array):
        return data


class SimpleStandardScaler:
    def __init__(self, data: np.array, one_interval: bool = False):
        self.mean = data.mean(axis=0)
        if one_interval:
            self.std = np.abs(data).max(axis=0)
        else:
            self.std = data.std(axis=0)

    def transform(self, data: np.array):
        return (data - self.mean) / self.std

    def inverse(self, data: np.array):
        return data * self.std + self.mean


class SimpleMinMaxScaler:
    def __init__(self, data: np.array):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data: np.array):
        return (data - self.min) / (self.max - self.min)

    def inverse(self, data: np.array):
        return data * (self.max - self.min) + self.min
