from typing import Optional

import numpy as np
from models.base import BaseUTSFModel
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf


class M4PretrainedNaiveModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.submission = kwargs["submissions"]["Naive"]

    def fit(self, data: np.array) -> None:
        pass

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return np.zeros((horizon,))
        return self.submission


class M4PretrainedSNaiveModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.submission = kwargs["submissions"]["sNaive"]

    def fit(self, data: np.array) -> None:
        pass

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return np.zeros((horizon,))
        return self.submission


class M4PretrainedNaive2Model(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.submission = kwargs["submissions"]["Naive2"]

    def fit(self, data: np.array) -> None:
        pass

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return np.zeros((horizon,))
        return self.submission
