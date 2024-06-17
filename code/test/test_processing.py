import sys
import os

sys.path.extend([os.path.abspath('.')])
sys.path.extend([os.path.abspath('./code')])

import os
import shutil
import tarfile
import pytest
import tempfile
import pandas as pd
from constants import *
from pathlib import Path
import subprocess


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    data_directory = Path(directory) / "data" 
    data_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_directory / "data.csv")
    
    directory = Path(directory)

    test_args = ['--pc_base_directory', f"{directory}",]

    subprocess.run(['python3', 
                    '/home/mahmood/ml-penguins-classification/code/processing.py'] + 
                    test_args).check_returncode()

    yield directory
    
    shutil.rmtree(directory)


def test_preprocess_generates_data_splits(directory):
    output_directories = os.listdir(directory)
    
    assert "train" in output_directories
    assert "test" in output_directories


def test_preprocess_generates_baselines(directory):
    output_directories = os.listdir(directory)
    assert "baseline" in output_directories
    assert "test-baseline.csv" in os.listdir(directory / "baseline")
    assert "train-baseline.csv" in os.listdir(directory / "baseline")


def test_preprocess_creates_two_models(directory):
    model_path = directory / "transformers"
    tar = tarfile.open(model_path / 'transformers.tar.gz', "r:gz")
    assert "transformers.joblib" in tar.getnames()


def test_splits_are_transformed(directory):
    X_train = pd.read_csv(directory / "train" / "X_train.csv", header=None)
    y_train = pd.read_csv(directory / "train" / "y_train.csv", header=None)
    X_test = pd.read_csv(directory / "test" / "X_test.csv", header=None)
    y_test = pd.read_csv(directory / "test" / "y_test.csv", header=None)

    # After transforming the data, the number of columns should be 7 for X data:
    # * 3 - island (one-hot encoded)
    # * 1 - culmen_length_mm = 1
    # * 1 - culmen_depth_mm
    # * 1 - flipper_length_mm
    # * 1 - body_mass_g
    # * 2 - sex (one-hot encoded)
    X_nr_columns = 9

    # After transforming the data, number of columns should be 3 for y data:
    # * 3 - species (one-hot encoded)
    y_nr_columns = 3

    assert X_train.shape[1] == X_nr_columns
    assert y_train.shape[1] == y_nr_columns
    assert X_test.shape[1] == X_nr_columns
    assert y_test.shape[1] == y_nr_columns


def test_train_baseline_is_not_transformed(directory):
    baseline = pd.read_csv(directory / "baseline" / "train-baseline.csv", header=None)

    island = baseline.iloc[:, 1].unique()

    assert "Biscoe" in island
    assert "Torgersen" in island
    assert "Dream" in island


def test_test_baseline_is_not_transformed(directory):
    baseline = pd.read_csv(directory / "baseline" / "test-baseline.csv", header=None)

    island = baseline.iloc[:, 1].unique()

    assert "Biscoe" in island
    assert "Torgersen" in island
    assert "Dream" in island