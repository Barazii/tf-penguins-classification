import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def process(pc_base_directory):
    df = _read_csv_data(pc_base_directory)
    train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=1)
    _save_baseline(pc_base_directory, train_df, test_df)

    # build the the column transformers
    transformers = ColumnTransformer(
        transformers=[
            ("numeric_tf", StandardScaler(), make_column_selector(dtype_include="float64")),
            ("categorical_tf", OneHotEncoder(), make_column_selector(dtype_include=object))
        ],
        verbose=True,
    )
    y_train = transformers.fit_transform(pd.DataFrame(train_df.species))
    y_test = transformers.transform(pd.DataFrame(test_df.species))
    train_df = train_df.drop(axis=1, columns="species")
    test_df = test_df.drop(axis=1, columns="species")
    X_train = transformers.fit_transform(train_df)
    X_test = transformers.transform(test_df)

    _save_splits(pc_base_directory, X_train, X_test, y_train, y_test)
    _save_transformers()

def _read_csv_data(pc_base_directory):
    csv_file_path = Path(os.path.join(pc_base_directory, "data")).glob("*.csv")
    df = [pd.read_csv(csv_file) for csv_file in csv_file_path].pop()
    return df.sample(axis=0, frac=1)

def _save_baseline(pc_base_directory, train_df, test_df):
    path = Path(pc_base_directory) / "baseline"
    path.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(path_or_buf=path/f"train-baseline.csv", index=False, header=False)
    test_df.to_csv(path_or_buf=path/f"test-baseline.csv", index=False, header=False)

def _save_splits(pc_base_directory, X_train, X_test, y_train, y_test):
    train_path = Path(pc_base_directory) / "train"
    test_path = Path(pc_base_directory) / "test"

    train_path.mkdir(exist_ok=True, parents=True)
    test_path.mkdir(exist_ok=True, parents=True)

    pd.DataFrame(X_train).to_csv(train_path/"X_train.csv", index=False, header=False)
    pd.DataFrame(y_train).to_csv(train_path/"y_train.csv", index=False, header=False)
    pd.DataFrame(X_test).to_csv(test_path/"X_test.csv", index=False, header=False)
    pd.DataFrame(y_test).to_csv(test_path/"y_test.csv", index=False, header=False)

def _save_transformers():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc_base_directory", type=str, required=True)
    args = parser.parse_args()
    process(args.pc_base_directory)