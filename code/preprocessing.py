import argparse
import pandas as pd
from ..constants import *
from sklearn.model_selection import train_test_split
from pathlib import Path


def preprocess():
    df = _read_csv_data()
    train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=1)
    _save_baseline(train_df, test_df)

def _read_csv_data():
    csv_file = os.listdir(input_data_directory).pop()
    df = pd.read_csv(csv_file)
    return df.sample(axis=0, frac=1)

def _save_baseline(train_df, test_df):
    path = Path(baseline_directory)
    path.mkdir(exist_ok=True, parents=True)
    train_df.to_csv(path_or_buf=path/f"train-baseline.csv", index=False, header=False)
    test_df.to_csv(path_or_buf=path/f"test-baseline.csv", index=False, header=False)


if __name__ == "__main__":
    preprocess()