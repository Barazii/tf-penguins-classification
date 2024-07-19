import sys
import os

sys.path.extend([os.path.abspath('./code')])
sys.path.extend([os.path.abspath('.')])

from postprocessing_component import input_fn, model_fn, output_fn, predict_fn
from constants import TARGET_CATEGORIES, CLEANED_DATA_PATH
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
        train_data_dir=pc_base_directory / "data-splits",
        hp_epochs=1,
    )

    yield pc_base_directory

    shutil.rmtree(pc_base_directory)

def test_model_fn():
    ret = model_fn()
    assert ret == TARGET_CATEGORIES

def test_input_fn(directory):
    model = models.load_model(directory / "model" / "assets")
    content_type = "application/json"
    sample_data = pd.read_csv(directory / "data-splits" / "test" / "X_test.csv")
    sample_data = sample_data.iloc[1:5,:]
    probabilities = model.predict(np.array(sample_data))
    model_output_probabilities = json.dumps({
        "predictions": probabilities.tolist(),
    })
    ret = input_fn(model_output_probabilities, content_type)
    assert ret == probabilities.tolist()

def test_input_fn_raise_error():
    content_type = "xyz"
    model_output_probabilities = None
    with pytest.raises(ValueError):
        input_fn(model_output_probabilities, content_type)

def test_predict_fn(directory):
    model = models.load_model(directory / "model" / "assets")
    content_type = "application/json"
    sample_data = pd.read_csv(directory / "data-splits" / "test" / "X_test.csv")
    sample_data = sample_data.iloc[1:5,:]
    probabilities = model.predict(np.array(sample_data))
    ret = predict_fn(probabilities, TARGET_CATEGORIES)
    assert len(ret) == len(sample_data)
    assert len(ret[0]) == 2
    assert isinstance(ret[0][0], str)
    assert isinstance(ret[0][1], np.float32)

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

def test_output_fn_json_accept():
    accept = "application/json"
    predictions = [(1,0.1), (2,0.2), (3,0.3)]
    output = [{"prediction": 1, "confidence": 0.1},
              {"prediction": 2, "confidence": 0.2},
              {"prediction": 3, "confidence": 0.3}]
    ret = output_fn(predictions, accept)
    assert ret[0][0] == output[0]
    assert ret[0][1] == output[1]
    assert ret[0][2] == output[2]
    assert ret[1] == accept