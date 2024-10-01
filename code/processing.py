import argparse
import pandas as pd
import os
from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tempfile
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
import tarfile
import json
import numpy as np


def process(pc_base_directory):
    df = _read_csv_data(pc_base_directory)
    X = df.drop(axis=1, columns="species")
    y = df.species
    
    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_validation_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41)
    for train_index, test_index in train_test_split.split(X, y):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

    for train_index, valid_index in train_validation_split.split(X_train_full, y_train_full):
        X_train, X_valid = X_train_full.iloc[train_index], X_train_full.iloc[valid_index]
        y_train, y_valid = y_train_full.iloc[train_index], y_train_full.iloc[valid_index]

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    validation_data = pd.concat([X_valid, y_valid], axis=1)
    _save_baseline(pc_base_directory, train_data, test_data, validation_data)

    # build the the column transformers
    transformers = ColumnTransformer(
        transformers=[
            ("numeric_tf", StandardScaler(), make_column_selector(dtype_include="float64")),
            ("categorical_tf", OneHotEncoder(), make_column_selector(dtype_include=object))
        ],
        verbose=True,
    )

    ty_train = transformers.fit_transform(pd.DataFrame(train_data.species))
    ty_validation = transformers.transform(pd.DataFrame(validation_data.species))
    ty_test = transformers.transform(pd.DataFrame(test_data.species))

    train_data = train_data.drop("species", axis=1)
    validation_data = validation_data.drop("species", axis=1)
    test_data = test_data.drop("species", axis=1)

    tX_train = transformers.fit_transform(train_data)
    tX_test = transformers.transform(test_data)
    tX_validation = transformers.transform(validation_data)

    t_train_data = pd.DataFrame(np.concatenate([tX_train, ty_train], axis=1))
    t_test_data = pd.DataFrame(np.concatenate([tX_test, ty_test], axis=1))
    t_validation_data = pd.DataFrame(np.concatenate([tX_validation, ty_validation], axis=1))
    _save_transformed_data(pc_base_directory, t_train_data, t_test_data, t_validation_data)
    _save_transformers(pc_base_directory, transformers)

def _read_csv_data(pc_base_directory):
    csv_file_path = (pc_base_directory / "data").glob("*.csv")
    df = [pd.read_csv(csv_file) for csv_file in csv_file_path].pop()
    return df.sample(axis=0, frac=1)

def _save_baseline(pc_base_directory, train_data, test_data, validation_data):
    path = pc_base_directory / "baseline"
    path.mkdir(exist_ok=True, parents=True)
    train_data.to_csv(path / "train-baseline.csv", header=True, index=False)
    test_data.to_csv(path / "test-baseline.csv", header=True, index=False)
    validation_data.to_csv(path / "validation-baseline.csv", header=True, index=False)

def _save_transformed_data(pc_base_directory, t_train_data, t_test_data, t_validation_data):
    path = pc_base_directory / "transformed-data"
    path.mkdir(exist_ok=True, parents=True)
    t_train_data.to_csv(path / "t_train_data.csv", header=False, index=False)
    t_test_data.to_csv(path / "t_test_data.csv", header=False, index=False)
    t_validation_data.to_csv(path / "t_validation_data.csv", header=False, index=False)

def _save_transformers(pc_base_directory, transformers):

    transformers_path = pc_base_directory / "transformers"
    transformers_path.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(transformers, os.path.join(directory, "transformers.joblib"))

        with tarfile.open(f"{str(transformers_path / 'transformers.tar.gz')}", "w:gz") as tar_file:
            tar_file.add(os.path.join(directory, "transformers.joblib"), arcname="transformers.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_base_directory", type=str, required=True)
    args = parser.parse_args()
    process(Path(args.pc_base_directory))