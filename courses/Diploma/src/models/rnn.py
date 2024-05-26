from typing import Optional

import numpy as np
import torch
from models.base import (BaseUTSFModel, SimpleDataset, SimpleMinMaxScaler,
                         SimpleNoScaleScaler, SimpleStandardScaler)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def replace_layers(model, old_layer, new_layer):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old_layer, new_layer)

        if isinstance(module, old_layer):
            setattr(model, n, new_layer)


class FCRNN(nn.Module):
    def __init__(self, rnn_model: str, input_size, hidden_size, num_layers, is_bidirectional, use_tanh, use_pi, fc_size, device):
        super().__init__()

        self.rnn_model = rnn_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        self.use_tanh = use_tanh
        self.use_pi = use_pi
        self.fc_size = fc_size
        self.device = device

        self.bi_mult = 2 if self.is_bidirectional else 1

        rnn_class = nn.RNN
        if self.rnn_model == "LSTM":
            rnn_class = nn.LSTM
        elif self.rnn_model == "GRU":
            rnn_class = nn.GRU
        elif self.rnn_model != "RNN":
            raise RuntimeError(f"Unknown RNN model type: {rnn_model}")

        self.rnn = rnn_class(input_size=self.input_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             bidirectional=self.is_bidirectional)
        if use_tanh:
            replace_layers(self.rnn, nn.Sigmoid, nn.Tanh)
        self.zero_memory()
        self.fc_1 = nn.Linear(self.bi_mult * self.hidden_size, self.fc_size)
        fc_add = 1 if self.use_pi else 0
        self.fc_2 = nn.Linear(self.fc_size + fc_add, 1)
        self.relu = nn.ReLU()

    def zero_memory(self) -> None:
        self.rnn_h = torch.zeros(self.bi_mult * self.num_layers, self.hidden_size).to(self.device)
        self.rnn_c = torch.zeros(self.bi_mult * self.num_layers, self.hidden_size).to(self.device)

    def forward(self, x):
        # print(f"in: {x.shape}")  # [input_size]
        # print(f"h, c: {self.rnn_h.shape}, {self.rnn_c.shape}")
        if self.rnn_model == "LSTM":
            out, (h, c) = self.rnn(x, (self.rnn_h, self.rnn_c))
            self.rnn_h, self.rnn_c = h.detach(), c.detach()
        else:
            out, h = self.rnn(x, self.rnn_h)
            self.rnn_h = h.detach()
        # print(f"RNN: {out.shape}")  #    [hidden_size]
        out = self.relu(out)
        out = self.fc_1(out)
        # print(f"FC 1: {out.shape}")  #    [fc_size]
        out = self.relu(out)
        if self.use_pi:
            out = torch.cat([out, x[:, :1]], dim=1)
        out = self.fc_2(out)
        # print(f"FC 2: {out.shape}")  #    [1]
        return out


class RNNUTSFSimpleModel(BaseUTSFModel):
    def __init__(self, **kwargs) -> None:
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.seasonality_period = kwargs.get("seasonality_period", None)
        self.input_size = 1 if self.seasonality_period is None else 2
        self.model = FCRNN(rnn_model=kwargs["rnn_model"],
                           input_size=self.input_size,
                           hidden_size=kwargs["hidden_size"],
                           num_layers=kwargs["num_layers"],
                           is_bidirectional=False,  # No sense
                           use_tanh=kwargs.get("use_tanh", False),
                           use_pi=kwargs.get("use_pi", False),
                           fc_size=kwargs["fc_size"],
                           device=self.device).to(self.device)
        self.learning_rate = kwargs["learning_rate"]
        self.weight_decay = kwargs["weight_decay"]
        self.num_epoch = kwargs["num_epoch"]
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate, weight_decay=self.weight_decay)
        scaler_classes = [SimpleStandardScaler, SimpleMinMaxScaler]
        self.scaler_class = SimpleNoScaleScaler
        for scaler in scaler_classes:
            if kwargs["scaler_class_name"] == scaler.__name__:
                self.scaler_class = scaler

    def fit(self, data: np.array) -> None:
        self.scaler = self.scaler_class(data)
        self.train_data = data
        if self.seasonality_period is not None:
            data_norm = data[self.seasonality_period:].reshape(1, -1)
            data_shift = np.roll(data, self.seasonality_period)[
                self.seasonality_period:].reshape(1, -1)
            self.train_data = np.concatenate([data_norm, data_shift])
            self.last_train_data = self.train_data[-self.seasonality_period:]
            data = self.train_data

        data = self.scaler.transform(data)
        X_train = data.reshape(-1, 1, self.input_size)[:-1]
        y_train = np.roll(data, -1).reshape(-1, 1, self.input_size)[:-1]
        X_train_tensor = torch.FloatTensor(X_train).to(self.device).requires_grad_()
        y_train_tensor = torch.FloatTensor(y_train).to(self.device).requires_grad_()
        self.train_dataloader = DataLoader(SimpleDataset(X_train_tensor, y_train_tensor),
                                           shuffle=False, batch_size=None)

        for epoch in range(self.num_epoch):
            self.model.train()
            self.model.zero_memory()
            for inputs, y_reg in tqdm(self.train_dataloader, desc=f'Training epoch {epoch + 1}', leave=False):
                self.optimizer.zero_grad()
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, y_reg)
                loss.backward()
                self.optimizer.step()

    def predict(self, horizon: int, data: Optional[np.array] = None) -> np.array:
        self.model.eval()

        if data is not None:
            if self.seasonality_period is not None:
                data_norm = data[self.seasonality_period:].reshape(1, -1)
                data_shift = np.roll(data, self.seasonality_period)[
                    self.seasonality_period:].reshape(1, -1)
                data = np.concatenate([data_norm, data_shift])
            data = self.scaler.transform(data).reshape(-1, 1, self.input_size)
            data_tensor = torch.FloatTensor(data).to(self.device)
            preds = []
            with torch.no_grad():
                self.model.zero_memory()
                for elem in data_tensor:
                    preds.append(self.model.forward(elem).cpu())

            preds = torch.cat(preds).view(-1).numpy()
            return self.scaler.inverse(preds)

        train_data = self.scaler.transform(self.train_data).reshape(-1, 1, self.input_size)
        train_data_tensor = torch.FloatTensor(train_data).to(self.device)
        with torch.no_grad():
            self.model.zero_memory()
            for elem in train_data_tensor:
                first_elem = self.model.forward(elem)

        preds = [first_elem]
        with torch.no_grad():
            for i in range(horizon - 1):
                preds.append(self.model.forward(preds[-1]))

        preds = torch.cat(preds).view(-1).cpu().numpy()
        return self.scaler.inverse(preds)
