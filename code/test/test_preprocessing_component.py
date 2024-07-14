import sys
import os

sys.path.extend([os.path.abspath('.')])
sys.path.extend([os.path.abspath('./code')])

from preprocessing_component import predict_fn, input_fn, output_fn, model_fn
from processing import process
import pytest
import tempfile
from pathlib import Path
import shutil
from constants import CLEANED_DATA_PATH, FEATURE_COLUMNS
from sklearn.compose import ColumnTransformer
import json


@pytest.fixture(scope="function", autouse=False)
def directory():
    temp_base_directory = tempfile.mkdtemp()
    temp_base_directory = Path(temp_base_directory)
    data_directory = Path(temp_base_directory) / "data" 
    data_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_directory / "data.csv")

    process(temp_base_directory)

    yield temp_base_directory
    
    shutil.rmtree(temp_base_directory)

def test_model_fn(directory):
    transformers = model_fn(directory / "transformers")
    assert isinstance(transformers, ColumnTransformer)

def test_input_fn_with_csv_content_type():
    input_content = """
        Torgersen,39.2,19.6,195,4675,MALE
    """
    input_content_type = "text/csv"
    df = input_fn(input_content, input_content_type)
    assert len(df.columns) == 6
    for col in df.columns.values:
        assert col in FEATURE_COLUMNS

def test_input_fn_with_json_content_type():
    input_content = json.dumps({
        "island": "Torgersen",
        "culmen_length_mm": 39.2,
        "culmen_depth_mm": 19.6,
        "flipper_length_mm": 195,
        "body_mass_g": 4675,
        "sex": "MALE",
    })
    input_content_type = "application/json"
    df = input_fn(input_content, input_content_type)
    assert len(df.columns) == 6
    for col in df.columns.values:
        assert col in FEATURE_COLUMNS