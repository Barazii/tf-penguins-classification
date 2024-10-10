import sys
import os

sys.path.extend([os.path.abspath('./code')])
sys.path.extend([os.path.abspath('.')])

from postprocessing_component import input_fn, model_fn, output_fn, predict_fn, TARGET_CATEGORIES
from constants import CLEANED_DATA_PATH
import tempfile
import pathlib
import shutil
from processing import process
from training import train
import pytest
from keras import models
import numpy as np
import pandas as pd
import json
import tarfile


@pytest.fixture(scope="function", autouse=False)
def directory(monkeypatch):
    pc_base_directory = tempfile.mkdtemp()
    pc_base_directory = pathlib.Path(pc_base_directory)
    data_dir = pc_base_directory / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_dir)

    process(pc_base_directory)

    # temporarily set the env variable SM_MODEL_DIR
    monkeypatch.setenv("SM_MODEL_DIR", f"{pc_base_directory}/model")
    train(
        train_data_dir=pc_base_directory / "transformed-data",
        valid_data_dir=pc_base_directory / "transformed-data",
        hp_epochs=1,
    )

    yield pc_base_directory

    shutil.rmtree(pc_base_directory)

def test_input_fn(directory):
    model_1 = models.load_model(directory / "model" / "001")
    content_type = "application/json"
    sample_data = pd.read_csv(directory / "transformed-data" / "t_train_data_1.csv", header=None)
    sample_data = sample_data.iloc[1:5,:-3]
    probabilities = model_1.predict(np.array(sample_data))
    model_output_probabilities = json.dumps({
        "predictions": probabilities.tolist(),
    })
    ret = input_fn(model_output_probabilities, content_type)
    assert ret == probabilities.tolist()

def test_input_fn_raise_error_about_content_type():
    content_type = "xyz"
    model_output_probabilities = [1,2]
    with pytest.raises(ValueError):
        input_fn(model_output_probabilities, content_type)

def test_input_fn_raise_error_about_model_predictions_1():
    content_type = "application/json"
    model_output_probabilities = None
    with pytest.raises(ValueError):
        input_fn(model_output_probabilities, content_type)

def test_input_fn_raise_error_about_model_predictions_2():
    content_type = "application/json"
    model_output_probabilities = []
    with pytest.raises(ValueError):
        input_fn(model_output_probabilities, content_type)

def test_predict_fn(directory):
    model = models.load_model(directory / "model" / "001")
    sample_data = pd.read_csv(directory / "transformed-data" / "t_train_data_1.csv", header=None)
    sample_data = sample_data.iloc[1:5,:-3]
    probabilities = model.predict(np.array(sample_data))
    ret = predict_fn(probabilities, TARGET_CATEGORIES)
    assert len(ret) == len(sample_data)
    assert len(ret[0]) == 2
    assert isinstance(ret[0][0], str)

def test_output_fn_raise_error():
    accept = "x"
    predictions = []
    with pytest.raises(ValueError):
        output_fn(predictions, accept)

def test_output_fn_csv_accept():
    accept = "text/csv"
    predictions = [1, 2, 3]
    ret = output_fn(predictions, accept)
    assert ret[0] == predictions
    assert ret[1] == accept

def test_output_fn_json_accept_multiple_predictions():
    accept = "application/json"
    predictions = [("Adelie",0.1), ("Chinstrap",0.2), ("Gentoo",0.3)]
    ret = output_fn(predictions, accept)
    assert ret[0][0] == {'prediction': 'Adelie', 'confidence': 0.1}
    assert ret[0][1] == {"prediction": "Chinstrap", "confidence": 0.2}
    assert ret[0][2] == {"prediction": "Gentoo", "confidence": 0.3}
    assert ret[1] == accept

def test_output_fn_json_accept_single_predictions():
    accept = "application/json"
    predictions = [("Adelie",0.1)]
    ret = output_fn(predictions, accept)
    assert ret[0] == {"prediction": "Adelie", "confidence": 0.1}
    assert ret[1] == accept