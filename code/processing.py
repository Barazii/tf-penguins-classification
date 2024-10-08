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
    n_splits = 3
    train_validation_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=41)

    for train_index, test_index in train_test_split.split(X, y):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data = []
    validation_data = []
    for i, (train_index, valid_index) in enumerate(train_validation_split.split(X_train_full, y_train_full)):
        X_train, X_valid = X_train_full.iloc[train_index], X_train_full.iloc[valid_index]
        y_train, y_valid = y_train_full.iloc[train_index], y_train_full.iloc[valid_index]
        train_data.append(pd.concat([X_train, y_train], axis=1))
        validation_data.append(pd.concat([X_valid, y_valid], axis=1))

    _save_baseline(pc_base_directory, train_data, test_data, validation_data)

    # build the the column transformers
    transformers = ColumnTransformer(
        transformers=[
            ("numeric_tf", StandardScaler(), make_column_selector(dtype_include="float64")),
            ("categorical_tf", OneHotEncoder(), make_column_selector(dtype_include=object))
        ],
        verbose=True,
    )

    t_train_data = []
    t_validation_data = []
    trans_list = []
    for (id, t_data, v_data) in zip(range(n_splits), train_data, validation_data):
        ty_train = transformers.fit_transform(pd.DataFrame(t_data.species))
        ty_validation = transformers.transform(pd.DataFrame(v_data.species))
        tX_train = transformers.fit_transform(t_data.drop("species", axis=1))
        tX_validation = transformers.transform(v_data.drop("species", axis=1))
        trans_list.append(transformers)
        t_train_data.append(pd.DataFrame(np.concatenate([tX_train, ty_train], axis=1)))
        t_validation_data.append(pd.DataFrame(np.concatenate([tX_validation, ty_validation], axis=1)))

    _save_transformed_data(pc_base_directory, t_train_data, t_validation_data)
    _save_transformers(pc_base_directory, trans_list)


def _read_csv_data(pc_base_directory):
    csv_file_path = (pc_base_directory / "data").glob("*.csv")
    df = [pd.read_csv(csv_file) for csv_file in csv_file_path].pop()
    return df.sample(axis=0, frac=1)

def _save_baseline(pc_base_directory, train_data, test_data, validation_data):
    path = pc_base_directory / "baseline"
    path.mkdir(exist_ok=True, parents=True)
    test_data.to_csv(path / "test-baseline.csv", header=True, index=False)
    for i,data in enumerate(train_data):
        data.to_csv(path / f"train-baseline-{i+1}.csv", header=True, index=False)
    for i,data in enumerate(validation_data):    
        data.to_csv(path / f"validation-baseline-{i+1}.csv", header=True, index=False)

def _save_transformed_data(pc_base_directory, t_train_data, t_validation_data):
    path = pc_base_directory / "transformed-data"
    path.mkdir(exist_ok=True, parents=True)
    for i,data in zip(range(len(t_train_data)), t_train_data):
        data.to_csv(path / f"t_train_data_{i+1}.csv", header=False, index=False)
    for i,data in zip(range(len(t_validation_data)), t_validation_data):
        data.to_csv(path / f"t_validation_data_{i+1}.csv", header=False, index=False)

def _save_transformers(pc_base_directory, transformers_list):

    transformers_path = pc_base_directory / "transformers"
    transformers_path.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as directory:
        for i,transformers in zip(range(len(transformers_list)), transformers_list):
            joblib.dump(transformers, os.path.join(directory, f"transformers_{i+1}.joblib"))

        with tarfile.open(transformers_path / 'transformers.tar.gz', "w:gz") as tar_file:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                tar_file.add(file_path, arcname=filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_base_directory", type=str, required=True)
    args = parser.parse_args()
    process(Path(args.pc_base_directory))