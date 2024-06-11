import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess(args):
    df = _read_csv_data()
    train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=1)
    _save_baseline(train_df, test_df)

    # build the the column transformers
    transformers = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler, make_column_selector(dtype_include="float64")),
            ("categorical", OneHotEncoder, make_column_selector(dtype_include=object))
        ],
        verbose=True,
    )
    y_train = transformers.fit_transform(train_df.species.values)
    y_test = transformers.transform(test_df.species.values)
    train_df.drop(axis=1, columns="species")
    test_df.drop(axis=1, columns="species")
    X_train = transformers.fit_transform(train_df)
    X_test = transformers.transform(test_df)

    _save_splits(X_train, X_test, y_train, y_test)

def _read_csv_data():
    csv_file = os.listdir(args.input_data_directory).pop()
    df = pd.read_csv(csv_file)
    return df.sample(axis=0, frac=1)

def _save_baseline(train_df, test_df):
    path = Path(args.baseline_directory)
    path.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(path_or_buf=path/f"train-baseline.csv", index=False, header=False)
    test_df.to_csv(path_or_buf=path/f"test-baseline.csv", index=False, header=False)

def _save_splits(X_train, X_test, y_train, y_test):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_directory", type=str, required=True)
    parser.add_argument("--data_splits_directory", type=str, required=True)
    parser.add_argument("--model_directory", type=str, required=True)
    parser.add_argument("--transformers_directory", type=str, required=True)
    parser.add_argument("--baseline_directory", type=str, required=True)
    args = parser.parse_args()
    preprocess(args)