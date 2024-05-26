from typing import Optional

import numpy as np
from models.base import BaseUTSFModel
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf


class M4NaiveModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        pass

    def fit(self, data: np.array) -> None:
        self.last_train = data[-1]

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return data
        return np.full((horizon,), self.last_train)


class M4NaiveSeasonalModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.seasonality_period = kwargs["seasonality_period"]

    def fit(self, data: np.array) -> None:
        self.last_train_data = data[-self.seasonality_period:]

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return data
        times = (horizon - 1) // self.seasonality_period + 1
        return np.tile(self.last_train_data, times)[:horizon]


class M4Naive2Model(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.seasonality_period = kwargs["seasonality_period"]

    def fit(self, data: np.array) -> None:
        self.data_train = data

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return data
        if self.seasonality_period <= 1 or not seasonality_test(self.data_train, self.seasonality_period):
            return np.full((horizon,), self.data_train[-1])
        dec = seasonal_decompose(self.data_train, model="multilplicative",
                                 period=self.seasonality_period).seasonal
        dec_last = (self.data_train / dec)[-1]

        times = (horizon - 1) // self.seasonality_period + 1
        seas_mult = np.repeat(dec[-self.seasonality_period:], times)
        return seas_mult[:horizon] * dec_last


def seasonality_test(input: np.array, ppy: int) -> bool:
    """Used to determine whether a time series is seasonal"""
    tcrit = norm.ppf(0.95)
    if len(input) < 3 * ppy:
        return False
    else:
        xacf = acf(input, nlags=ppy)[1:]
        clim = tcrit / np.sqrt(len(input)) * \
            np.sqrt(np.cumsum(np.concatenate([[1], 2 * xacf ** 2])))
        test_seasonal = np.abs(xacf[ppy - 1]) > clim[ppy - 1]
        return test_seasonal
