from typing import Any


def get_model_kwargs_by_name(model_name: str, **kwargs_outer) -> dict[str, Any]:
    result_kwargs = {}

    if model_name == "M4NaiveModel":
        pass
    elif model_name == "M4NaiveSeasonalModel":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
    elif model_name == "M4Naive2Model":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
    elif model_name == "M4PretrainedNaiveModel":
        result_kwargs["submissions"] = kwargs_outer["data_entity"]["submissions"]
    elif model_name == "M4PretrainedSNaiveModel":
        result_kwargs["submissions"] = kwargs_outer["data_entity"]["submissions"]
    elif model_name == "M4PretrainedNaive2Model":
        result_kwargs["submissions"] = kwargs_outer["data_entity"]["submissions"]
    elif model_name == "ARMAAutotuningModel_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["p_range"] = range(0, 3)
        result_kwargs["d_range"] = [0]
        result_kwargs["q_range"] = range(0, 3)
    elif model_name == "ARIMAAutotuningModel_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["p_range"] = range(0, 3)
        result_kwargs["d_range"] = range(1, 3)
        result_kwargs["q_range"] = range(0, 3)
    elif model_name == "SARIMAAutotuningModel_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["p_range"] = range(0, 2)
        result_kwargs["d_range"] = [1]
        result_kwargs["q_range"] = range(0, 2)
        result_kwargs["P_range"] = range(0, 2)
        result_kwargs["D_range"] = range(0, 2)
        result_kwargs["Q_range"] = range(0, 2)
    elif model_name == "TBATSModel_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
    elif model_name == "RNNUTSFSimpleModel_v1":
        result_kwargs["rnn_model"] = "RNN"
        result_kwargs["scaler_class_name"] = "SimpleNoScaleScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "LSTMUTSFSimpleModel_v1":
        result_kwargs["rnn_model"] = "LSTM"
        result_kwargs["scaler_class_name"] = "SimpleNoScaleScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "GRUUTSFSimpleModel_v1":
        result_kwargs["rnn_model"] = "GRU"
        result_kwargs["scaler_class_name"] = "SimpleNoScaleScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "RNNUTSFSimpleModelScaled_v1":
        result_kwargs["rnn_model"] = "RNN"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "LSTMUTSFSimpleModelScaled_v1":
        result_kwargs["rnn_model"] = "LSTM"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "GRUUTSFSimpleModelScaled_v1":
        result_kwargs["rnn_model"] = "GRU"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "RNNUTSFSimpleModelScaled_v2":
        result_kwargs["rnn_model"] = "RNN"
        result_kwargs["scaler_class_name"] = "SimpleMinMaxScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "LSTMUTSFSimpleModelScaled_v2":
        result_kwargs["rnn_model"] = "LSTM"
        result_kwargs["scaler_class_name"] = "SimpleMinMaxScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "GRUUTSFSimpleModelScaled_v2":
        result_kwargs["rnn_model"] = "GRU"
        result_kwargs["scaler_class_name"] = "SimpleMinMaxScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "RNNUTSFSimpleModelScaledTanh_v1":
        result_kwargs["rnn_model"] = "RNN"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_tanh"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "LSTMUTSFSimpleModelScaledTanh_v1":
        result_kwargs["rnn_model"] = "LSTM"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_tanh"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "GRUUTSFSimpleModelScaledTanh_v1":
        result_kwargs["rnn_model"] = "GRU"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_tanh"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "RNNUTSFSimpleModelScaledSeasonal_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["rnn_model"] = "RNN"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "LSTMUTSFSimpleModelScaledSeasonal_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["rnn_model"] = "LSTM"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "GRUUTSFSimpleModelScaledSeasonal_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["rnn_model"] = "GRU"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "RNNUTSFSimpleModelScaledPI_v1":
        result_kwargs["rnn_model"] = "RNN"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_pi"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "LSTMUTSFSimpleModelScaledPI_v1":
        result_kwargs["rnn_model"] = "LSTM"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_pi"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "GRUUTSFSimpleModelScaledPI_v1":
        result_kwargs["rnn_model"] = "GRU"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_pi"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "RNNUTSFSimpleModelScaledSeasonalPI_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["rnn_model"] = "RNN"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_pi"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "LSTMUTSFSimpleModelScaledSeasonalPI_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["rnn_model"] = "LSTM"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_pi"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    elif model_name == "GRUUTSFSimpleModelScaledSeasonalPI_v1":
        result_kwargs["seasonality_period"] = int(
            kwargs_outer["data_entity"]["meta"]["seasonality_period"])
        result_kwargs["rnn_model"] = "GRU"
        result_kwargs["scaler_class_name"] = "SimpleStandardScaler"
        result_kwargs["hidden_size"] = 32
        result_kwargs["num_layers"] = 1
        result_kwargs["use_pi"] = True
        result_kwargs["fc_size"] = 8
        result_kwargs["learning_rate"] = 0.003
        result_kwargs["weight_decay"] = 0.0004
        result_kwargs["num_epoch"] = 15
    else:
        raise RuntimeError(f"No such config for model type: {model_name}")

    return result_kwargs
