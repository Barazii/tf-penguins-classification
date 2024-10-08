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
from processing import process


@pytest.fixture(scope="function", autouse=False)
def directory():
    directory = tempfile.mkdtemp()
    data_directory = Path(directory) / "data" 
    data_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_directory / "data.csv")
    
    directory = Path(directory)

    process(directory)

    yield directory
    
    shutil.rmtree(directory)


def test_preprocess_generates_transformed_data(directory):
    output_directories = os.listdir(directory)
    
    assert "transformed-data" in output_directories
    assert "t_train_data_1.csv" in os.listdir(directory / "transformed-data")
    assert "t_train_data_2.csv" in os.listdir(directory / "transformed-data")
    assert "t_train_data_3.csv" in os.listdir(directory / "transformed-data")
    assert "t_validation_data_1.csv" in os.listdir(directory / "transformed-data")
    assert "t_validation_data_2.csv" in os.listdir(directory / "transformed-data")
    assert "t_validation_data_3.csv" in os.listdir(directory / "transformed-data")



def test_preprocess_generates_baselines(directory):
    output_directories = os.listdir(directory)
    assert "baseline" in output_directories
    assert "test-baseline.csv" in os.listdir(directory / "baseline")
    assert "train-baseline-1.csv" in os.listdir(directory / "baseline")
    assert "train-baseline-2.csv" in os.listdir(directory / "baseline")
    assert "train-baseline-3.csv" in os.listdir(directory / "baseline")
    assert "validation-baseline-1.csv" in os.listdir(directory / "baseline")
    assert "validation-baseline-2.csv" in os.listdir(directory / "baseline")
    assert "validation-baseline-3.csv" in os.listdir(directory / "baseline")


def test_preprocess_creates_data_transformers(directory):
    path = directory / "transformers"
    tar = tarfile.open(path / 'transformers.tar.gz', "r:gz")
    assert "transformers_1.joblib" in tar.getnames()
    assert "transformers_2.joblib" in tar.getnames()
    assert "transformers_3.joblib" in tar.getnames()


def test_splits_are_transformed(directory):
    train_data = pd.read_csv(directory / "transformed-data" / "t_train_data_1.csv", header=None)
    validation_data = pd.read_csv(directory / "transformed-data" / "t_validation_data_3.csv", header=None)

    # After transforming the data, the number of columns should be 
    # 12 for X data and y:
    # * 3 - island (one-hot encoded)
    # * 3 - species (one-hot encoded)
    # * 1 - culmen_length_mm = 1
    # * 1 - culmen_depth_mm
    # * 1 - flipper_length_mm
    # * 1 - body_mass_g
    # * 2 - sex (one-hot encoded)
    nr_columns = 12

    assert train_data.shape[1] == nr_columns
    assert validation_data.shape[1] == nr_columns


def test_baseline_data_is_not_transformed(directory):
    baseline = pd.read_csv(directory / "baseline" / "train-baseline-1.csv", header=None)

    island = baseline.iloc[:, 0].unique()
    assert "Biscoe" in island
    assert "Torgersen" in island
    assert "Dream" in island

    baseline = pd.read_csv(directory / "baseline" / "test-baseline.csv", header=None)

    island = baseline.iloc[:, 0].unique()
    assert "Biscoe" in island
    assert "Torgersen" in island
    assert "Dream" in island

    baseline = pd.read_csv(directory / "baseline" / "validation-baseline-2.csv", header=None)

    island = baseline.iloc[:, 0].unique()
    assert "Biscoe" in island
    assert "Torgersen" in island
    assert "Dream" in island