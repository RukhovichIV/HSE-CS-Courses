import os
import pathlib
import pickle
import random
from time import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from configs import get_model_kwargs_by_name
from metrics import compute_metrics
from models.arima import (ARIMAAutotuningModel, SARIMAAutotuningModel,
                          TBATSModel)
from models.base import BaseUTSFModel
from models.naive import M4Naive2Model, M4NaiveModel, M4NaiveSeasonalModel
from models.pretrained import (M4PretrainedNaive2Model, M4PretrainedNaiveModel,
                               M4PretrainedSNaiveModel)
from models.rnn import RNNUTSFSimpleModel
from tqdm import tqdm


def train_models_and_compute_metrics() -> None:
    def dataset_path_generator(name):
        return pathlib.Path.cwd().parent / "src" / "data" / f"{name}" / "dataset.pickle"
    dataset_names = ["m4"]
    model_types: list[BaseUTSFModel] = [
        (M4NaiveModel, "M4NaiveModel"),
        (M4NaiveSeasonalModel, "M4NaiveSeasonalModel"),
        (M4Naive2Model, "M4Naive2Model"),
        (M4PretrainedNaiveModel, "M4PretrainedNaiveModel"),
        (M4PretrainedSNaiveModel, "M4PretrainedSNaiveModel"),
        (M4PretrainedNaive2Model, "M4PretrainedNaive2Model"),
        (ARIMAAutotuningModel, "ARMAAutotuningModel_v1"),
        (ARIMAAutotuningModel, "ARIMAAutotuningModel_v1"),
        (SARIMAAutotuningModel, "SARIMAAutotuningModel_v1"),
        (TBATSModel, "TBATSModel_v1"),
        (RNNUTSFSimpleModel, "RNNUTSFSimpleModel_v1"),
        (RNNUTSFSimpleModel, "LSTMUTSFSimpleModel_v1"),
        (RNNUTSFSimpleModel, "GRUUTSFSimpleModel_v1"),
        (RNNUTSFSimpleModel, "RNNUTSFSimpleModelScaled_v1"),
        (RNNUTSFSimpleModel, "LSTMUTSFSimpleModelScaled_v1"),
        (RNNUTSFSimpleModel, "GRUUTSFSimpleModelScaled_v1"),
        (RNNUTSFSimpleModel, "RNNUTSFSimpleModelScaled_v2"),
        (RNNUTSFSimpleModel, "LSTMUTSFSimpleModelScaled_v2"),
        (RNNUTSFSimpleModel, "GRUUTSFSimpleModelScaled_v2"),
        (RNNUTSFSimpleModel, "RNNUTSFSimpleModelScaledTanh_v1"),
        (RNNUTSFSimpleModel, "LSTMUTSFSimpleModelScaledTanh_v1"),
        (RNNUTSFSimpleModel, "GRUUTSFSimpleModelScaledTanh_v1"),
        (RNNUTSFSimpleModel, "RNNUTSFSimpleModelScaledPI_v1"),
        (RNNUTSFSimpleModel, "LSTMUTSFSimpleModelScaledPI_v1"),
        (RNNUTSFSimpleModel, "GRUUTSFSimpleModelScaledPI_v1"),
    ]
    metrics = {"train": set(["RMSE", "sMAPE"]),
               "test": set(["RMSE", "MASE", "sMAPE", "OWA"])}
    results_save_path = pathlib.Path.cwd().parent / "src" / "data" / "results.pickle"

    save_period_minutes = 10
    save_period_start = time()

    results = {}
    random_state = 43
    for model_class, model_codename in model_types:
        results[model_codename] = {}
        for dataset_name in dataset_names:
            data_path = dataset_path_generator(dataset_name)
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            for entity_name, data_entity in tqdm(data.items(), desc=f"Model {model_codename} on {dataset_name}"):
                # Fix experiment
                torch.manual_seed(random_state)
                random.seed(random_state)
                np.random.seed(random_state)

                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
                torch.cuda.empty_cache()

                experiment_name = f'{dataset_name}_{entity_name}_{data_entity["meta"]["data_category"]}_{data_entity["meta"]["period"]}'
                results[model_codename][experiment_name] = {}

                # Data and model init
                train_series = np.array(data_entity["train"])
                test_series = np.array(data_entity["test"])
                y_test = test_series
                input_kwargs = get_model_kwargs_by_name(model_codename, data_entity=data_entity)
                model = model_class(**input_kwargs)

                # Train + predict on train
                time_start = time()
                model.fit(train_series)
                time_fit = time() - time_start
                time_start = time()
                y_pred = model.predict(train_series.shape[0], train_series)
                time_predict = time() - time_start
                results[model_codename][experiment_name][f"train"] = compute_metrics(
                    train_series, y_pred, time_predict, metrics["train"],
                    time_fit=time_fit,
                    period_weight=data_entity["meta"]["period_weight"])

                # Predict on test
                time_start = time()
                y_pred = model.predict(y_test.shape[0])
                time_predict = time() - time_start
                results[model_codename][experiment_name][f"test"] = compute_metrics(
                    y_test, y_pred, time_predict, metrics["test"],
                    X_train=train_series,
                    seasonality_period=data_entity["meta"]["seasonality_period"],
                    period_weight=data_entity["meta"]["period_weight"])

                # Save results
                if time() - save_period_start > save_period_minutes * 60:
                    with open(results_save_path, "wb") as f:
                        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                    save_period_start = time()

    with open(results_save_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def summarize_metrics(metrics_path: str) -> dict[str, Any]:
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    results = {}
    for model_codename, model_mtx in metrics.items():
        results[model_codename] = None
        for dataset_name, dataset_mtx in model_mtx.items():
            if results[model_codename] is None:
                results[model_codename] = {}
                for stage_name in dataset_mtx:
                    results[model_codename][stage_name] = {}
                    for metric_name, metric_value in dataset_mtx[stage_name].items():
                        results[model_codename][stage_name][metric_name] = metric_value
                        results[model_codename][stage_name][f"{metric_name}_mtcnt"] = 1
            else:
                for stage_name, stage_mtx in dataset_mtx.items():
                    for metric_name, metric_value in stage_mtx.items():
                        results[model_codename][stage_name][metric_name] += metric_value
                        results[model_codename][stage_name][f"{metric_name}_mtcnt"] += 1

        for stage_name in results[model_codename]:
            keys_to_delete = []
            for metric_name in results[model_codename][stage_name]:
                if metric_name.endswith("_mtcnt"):
                    continue
                results[model_codename][stage_name][
                    metric_name] /= results[model_codename][stage_name][f"{metric_name}_mtcnt"]
                keys_to_delete.append(f"{metric_name}_mtcnt")
            for key in keys_to_delete:
                del results[model_codename][stage_name][key]
    return results


def summarize_metrics_by_model(metrics_path: str, model_codename: str) -> dict[str, Any]:
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    model_mtx = metrics[model_codename]
    results = {}
    for dataset_name, dataset_mtx in model_mtx.items():
        cur_domain = dataset_name.split("_")[2]
        cur_frequency = dataset_name.split("_")[3]

        for stage_name in dataset_mtx:
            for metric_name, metric_value in dataset_mtx[stage_name].items():
                unit_name = f"{stage_name}_{metric_name}"
                if unit_name not in results:
                    results[unit_name] = {}
                    results[unit_name]["by_frequency"] = {}
                    results[unit_name]["by_frequency"][cur_frequency] = [metric_value, 1]
                    results[unit_name]["by_domain"] = {}
                    results[unit_name]["by_domain"][cur_domain] = [metric_value, 1]
                else:
                    if cur_frequency not in results[unit_name]["by_frequency"]:
                        results[unit_name]["by_frequency"][cur_frequency] = [metric_value, 1]
                    else:
                        results[unit_name]["by_frequency"][cur_frequency][0] += metric_value
                        results[unit_name]["by_frequency"][cur_frequency][1] += 1

                    if cur_domain not in results[unit_name]["by_domain"]:
                        results[unit_name]["by_domain"][cur_domain] = [metric_value, 1]
                    else:
                        results[unit_name]["by_domain"][cur_domain][0] += metric_value
                        results[unit_name]["by_domain"][cur_domain][1] += 1

    for metric_name in results:
        for frequency in results[metric_name]["by_frequency"]:
            lst = results[metric_name]["by_frequency"][frequency]
            results[metric_name]["by_frequency"][frequency] = lst[0] / lst[1]
        results[metric_name]["by_frequency"] = pd.Series(results[metric_name]["by_frequency"])

        for domain in results[metric_name]["by_domain"]:
            lst = results[metric_name]["by_domain"][domain]
            results[metric_name]["by_domain"][domain] = lst[0] / lst[1]
        results[metric_name]["by_domain"] = pd.Series(results[metric_name]["by_domain"])

    return results


def summarize_metrics_m4(metrics_path: str) -> dict[str, Any]:
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    results = {}
    for model_codename, model_mtx in metrics.items():
        results[model_codename] = None
        for dataset_name, dataset_mtx in model_mtx.items():
            if not dataset_name.startswith("m4_"):
                continue
            data_name, entity_name, category, period = dataset_name.split("_")
            if results[model_codename] is None:
                results[model_codename] = {}
            if category not in results[model_codename]:
                results[model_codename][category] = {}
            if period not in results[model_codename][category]:
                results[model_codename][category][period] = {}
            for stage_name, stage_mtx in dataset_mtx.items():
                if stage_name not in results[model_codename][category][period]:
                    results[model_codename][category][period][stage_name] = {}
                for metric_name, metric_value in stage_mtx.items():
                    if metric_name not in results[model_codename][category][period][stage_name]:
                        results[model_codename][category][period][stage_name][metric_name] = 0
                        results[model_codename][category][period][stage_name][f"{metric_name}_mtcnt"] = 0
                    results[model_codename][category][period][stage_name][metric_name] += metric_value
                    results[model_codename][category][period][stage_name][f"{metric_name}_mtcnt"] += 1

        for category in results[model_codename]:
            for period in results[model_codename][category]:
                for stage_name in results[model_codename][category][period]:
                    keys_to_delete = []
                    for metric_name in results[model_codename][category][period][stage_name]:
                        if metric_name.endswith("_mtcnt"):
                            continue
                        results[model_codename][category][period][stage_name][
                            metric_name] /= results[model_codename][category][period][stage_name][f"{metric_name}_mtcnt"]
                        keys_to_delete.append(f"{metric_name}_mtcnt")
                    for key in keys_to_delete:
                        del results[model_codename][category][period][stage_name][key]
    return results
