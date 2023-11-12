# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernel_info:
#     name: local-env
#   kernelspec:
#     display_name: Python 3.9.6 64-bit
#     language: python
#     name: python3
# ---

# %% jupyter={"outputs_hidden": false, "source_hidden": false} nteract={"transient": {"deleting": false}}
import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

import mlflow

# %%
TARGET_COL = "cost"

NUMERIC_COLS = [
    "distance",
    "dropoff_latitude",
    "dropoff_longitude",
    "passengers",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_weekday",
    "pickup_month",
    "pickup_monthday",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "dropoff_weekday",
    "dropoff_month",
    "dropoff_monthday",
    "dropoff_hour",
    "dropoff_minute",
    "dropoff_second",
]

CAT_NOM_COLS = [
    "store_forward",
    "vendor",
]

CAT_ORD_COLS = []


# %% jupyter={"outputs_hidden": false, "source_hidden": false} nteract={"transient": {"deleting": false}}
# Define Arguments for this step


class MyArgs:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)


args = MyArgs(
    raw_data="../../data/",
    train_data="/tmp/prep/train",
    val_data="/tmp/prep/val",
    test_data="/tmp/prep/test",
)

os.makedirs(args.train_data, exist_ok=True)
os.makedirs(args.val_data, exist_ok=True)
os.makedirs(args.test_data, exist_ok=True)


# %%


def main(args):
    """Read, split, and save datasets"""

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    print("mounted_path files: ")
    arr = os.listdir(args.raw_data)
    print(arr)

    data = pd.read_csv((Path(args.raw_data) / "taxi-data.csv"))
    data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]

    # ------------- Split Data ------------- #
    # -------------------------------------- #

    # Split data into train, val and test datasets

    random_data = np.random.rand(len(data))

    msk_train = random_data < 0.7
    msk_val = (random_data >= 0.7) & (random_data < 0.85)
    msk_test = random_data >= 0.85

    train = data[msk_train]
    val = data[msk_val]
    test = data[msk_test]

    mlflow.log_metric("train size", train.shape[0])
    mlflow.log_metric("val size", val.shape[0])
    mlflow.log_metric("test size", test.shape[0])

    train.to_parquet((Path(args.train_data) / "train.parquet"))
    val.to_parquet((Path(args.val_data) / "val.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))


# %% jupyter={"outputs_hidden": false, "source_hidden": false} nteract={"transient": {"deleting": false}}
mlflow.start_run()

lines = [
    f"Raw data path: {args.raw_data}",
    f"Train dataset output path: {args.train_data}",
    f"Val dataset output path: {args.val_data}",
    f"Test dataset path: {args.test_data}",
]

for line in lines:
    print(line)

main(args)

mlflow.end_run()

# %% jupyter={"outputs_hidden": false, "source_hidden": false} nteract={"transient": {"deleting": false}} vscode={"languageId": "shellscript"}
# ls "/tmp/prep/train"
