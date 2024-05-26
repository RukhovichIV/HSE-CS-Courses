import os
import pathlib
import pickle
import random
import subprocess

import pandas as pd
from dataloaders.utils import download_file
from tqdm import tqdm


def download_and_parse_dataset(force_download: bool = False) -> pathlib.Path:
    local_path_prefix = pathlib.Path.cwd().parent / "src" / "data" / "m4"
    dataset_local_path = local_path_prefix / "dataset.pickle"

    if force_download or not os.path.isfile(dataset_local_path):
        info_csv_url = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/M4-info.csv"
        periods = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
        pretrained_models = ["Naive", "Naive2", "sNaive"]

        def train_csv_url(period):
            return f"https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/{period}-train.csv"

        def test_csv_url(period):
            return f"https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Test/{period}-test.csv"

        def submissions_url(model):
            return f"https://github.com/Mcompetitions/M4-methods/raw/master/Point%20Forecasts/submission-{model}.rar"

        info_local_path = local_path_prefix / "raw" / "info.csv"

        def train_local_path(period):
            return local_path_prefix / "raw" / f"{period}_train.csv"

        def test_local_path(period):
            return local_path_prefix / "raw" / f"{period}_test.csv"

        def submissions_local_path(model, is_rar: bool = True):
            return local_path_prefix / "submissions" / f'{model}.csv{".rar" if is_rar else ""}'

        # Download if not present
        download_file(info_csv_url, info_local_path)
        for period in periods:
            download_file(train_csv_url(period), train_local_path(period))
            download_file(test_csv_url(period), test_local_path(period))
        # Original submissions
        for model in pretrained_models:
            download_file(submissions_url(model), submissions_local_path(model))
            subprocess.run(["unrar", "e",
                            submissions_local_path(model),
                            submissions_local_path(model).parent])
            subprocess.run(["mv",
                            submissions_local_path(model).parent / f"submission-{model}.csv",
                            submissions_local_path(model, is_rar=False)])

        # Read with pandas
        info_df = pd.read_csv(info_local_path)
        train_df, test_df = [], []
        norm_per_period = {}
        for period in tqdm(periods, desc="Reading data"):
            train_df.append(pd.read_csv(train_local_path(period), index_col="V1"))
            next_test_df = pd.read_csv(test_local_path(period), index_col="V1")
            norm_per_period[period] = 100000 / len(periods) / next_test_df.shape[0]
            test_df.append(next_test_df)
        train_df = pd.concat(train_df)
        test_df = pd.concat(test_df)
        pretrained_submissions = {}
        for model in pretrained_models:
            pretrained_submissions[model] = pd.read_csv(
                submissions_local_path(model, is_rar=False), index_col="id")

        # Make single dataset
        data = {}
        for idx in tqdm(info_df.index, desc="Making datasets"):
            row = info_df.iloc[idx]
            name = row["M4id"]
            data[name] = {}
            data[name]["train"] = train_df.loc[name].dropna().values
            data[name]["test"] = test_df.loc[name].dropna().values
            data[name]["meta"] = {}
            data[name]["meta"]["data_category"] = row["category"]
            data[name]["meta"]["period"] = row["SP"]
            data[name]["meta"]["date_start"] = row["StartingDate"]
            data[name]["meta"]["seasonality_period"] = row["Frequency"]
            data[name]["meta"]["period_weight"] = norm_per_period[row["SP"]]
            data[name]["submissions"] = {}
            for model in pretrained_models:
                data[name]["submissions"][model] = pretrained_submissions[model].loc[name].dropna().values

        # Save locally
        with open(dataset_local_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Return filepath
    return dataset_local_path


def make_random_dataset_subset(dataset_path: str, subset_path: str, subset_size: int):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    data_subset = {}
    subset_size = min(subset_size, len(data))
    subset_idxs = random.choices(list(data.keys()), k=subset_size)
    for idx in tqdm(subset_idxs):
        if len(data[idx]["train"]) > 1000:
            continue
        data_subset[idx] = data[idx]

    # Save locally
    with open(subset_path, "wb") as f:
        pickle.dump(data_subset, f, protocol=pickle.HIGHEST_PROTOCOL)
