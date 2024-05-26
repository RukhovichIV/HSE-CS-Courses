from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Optional

import numpy as np
from metrics import compute_sMAPE
from models.base import BaseUTSFModel
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tbats import TBATS


class ARIMAAutotuningModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.seasonality_period = kwargs["seasonality_period"]
        self.p_range = kwargs["p_range"]
        self.d_range = kwargs["d_range"]
        self.q_range = kwargs["q_range"]

    def fit(self, data: np.array) -> None:
        X = data[:-self.seasonality_period]
        y_true = data[-self.seasonality_period:]

        best_smape, best_params = float("inf"), None
        for p in self.p_range:
            for d in self.d_range:
                for q in self.q_range:
                    with redirect_stderr(StringIO()):
                        model = ARIMA(X, order=(p, d, q)).fit()
                        y_pred = model.forecast(self.seasonality_period)
                        cur_smape = compute_sMAPE(y_true, y_pred)
                        if cur_smape < best_smape:
                            best_smape = cur_smape
                            best_params = (p, d, q)

        with redirect_stderr(StringIO()):
            self.model = ARIMA(data, order=best_params).fit()

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return data
        return np.array(self.model.forecast(steps=horizon))


class SARIMAAutotuningModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.seasonality_period = kwargs["seasonality_period"]
        self.p_range = kwargs["p_range"]
        self.d_range = kwargs["d_range"]
        self.q_range = kwargs["q_range"]
        self.P_range = kwargs["P_range"]
        self.D_range = kwargs["D_range"]
        self.Q_range = kwargs["Q_range"]

    def fit(self, data: np.array) -> None:
        X = data[:-self.seasonality_period]
        y_true = data[-self.seasonality_period:]

        best_smape, best_params, best_params_seasonal = float("inf"), None, None
        s = self.seasonality_period
        for p in self.p_range:
            for d in self.d_range:
                for q in self.q_range:
                    with redirect_stderr(StringIO()):
                        if s > 1:
                            for P in self.P_range:
                                for D in self.D_range:
                                    for Q in self.Q_range:
                                        model = SARIMAX(X, order=(p, d, q),
                                                        seasonal_order=(P, D, Q, s)).fit(disp=False)
                                        y_pred = model.forecast(self.seasonality_period)
                                        cur_smape = compute_sMAPE(y_true, y_pred)
                                        if cur_smape < best_smape:
                                            best_smape = cur_smape
                                            best_params = (p, d, q)
                                            best_params_seasonal = (P, D, Q, s)
                        else:
                            model = ARIMA(X, order=(p, d, q)).fit()
                            y_pred = model.forecast(self.seasonality_period)
                            cur_smape = compute_sMAPE(y_true, y_pred)
                            if cur_smape < best_smape:
                                best_smape = cur_smape
                                best_params = (p, d, q)

        with redirect_stderr(StringIO()):
            if s > 1:
                self.model = SARIMAX(data, order=best_params,
                                     seasonal_order=best_params_seasonal).fit(disp=False)
            else:
                self.model = ARIMA(data, order=best_params).fit()

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return data
        return np.array(self.model.forecast(steps=horizon))


class TBATSModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.seasonality_period = kwargs["seasonality_period"]

    def fit(self, data: np.array) -> None:
        with redirect_stderr(StringIO()):
            self.model = TBATS(seasonal_periods=[self.seasonality_period]).fit(data)

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        if data is not None:
            return data
        return np.array(self.model.forecast(steps=horizon))
