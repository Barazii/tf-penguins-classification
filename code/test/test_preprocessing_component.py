import os
from dotenv import load_dotenv
from preprocessing_component import (
    predict_fn,
    input_fn,
    output_fn,
    model_fn,
    FEATURE_COLUMNS,
)
from processing import process
import pytest
import tempfile
from pathlib import Path
import shutil
from sklearn.compose import ColumnTransformer
import json
import tarfile


load_dotenv()


@pytest.fixture(scope="function", autouse=False)
def directory():
    temp_base_directory = tempfile.mkdtemp()
    temp_base_directory = Path(temp_base_directory)
    data_directory = Path(temp_base_directory) / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.environ["CLEANED_DATA_PATH"], data_directory / "data.csv")

    process(temp_base_directory)

    yield temp_base_directory

    shutil.rmtree(temp_base_directory)


def test_model_fn(directory):
    with tarfile.open(directory / "transformers" / "transformers.tar.gz") as tar_file:
        tar_file.extractall(path=directory / "transformers")
    transformers = model_fn(directory / "transformers")
    assert isinstance(transformers, ColumnTransformer)


def test_input_fn_with_empty_input_data():
    input_content = None
    input_content_type = "text/csv"
    with pytest.raises(ValueError):
        input_fn(input_content, input_content_type)


def test_input_fn_with_csv_content_type():
    input_content = """
        Torgersen,39.2,19.6,195,4675,MALE
    """
    input_content_type = "text/csv"
    df = input_fn(input_content, input_content_type)
    assert len(df.columns) == 6
    # make sure we didn't drop the wrong column
    assert "Adelie" not in df.values
    assert "Chinstrap" not in df.values
    assert "Gentoo" not in df.values


def test_input_fn_with_json_content_type():
    input_content = json.dumps(
        {
            "island": "Torgersen",
            "culmen_length_mm": 39.2,
            "culmen_depth_mm": 19.6,
            "flipper_length_mm": 195,
            "body_mass_g": 4675,
            "sex": "MALE",
        }
    )
    input_content_type = "application/json"
    df = input_fn(input_content, input_content_type)
    assert len(df.columns) == 6
    # make sure we didn't drop the wrong column
    assert "Adelie" not in df.values
    assert "Chinstrap" not in df.values
    assert "Gentoo" not in df.values


def test_input_fn_raise_error():
    input_content = "test"
    input_content_type = "wrong-content-type"
    with pytest.raises(ValueError):
        input_fn(input_content, input_content_type)


def test_predict_fn_try_route(directory):
    input_content = """
        Torgersen,39.2,19.6,195,4675,MALE
    """
    input_content_type = "text/csv"
    df = input_fn(input_content, input_content_type)

    with tarfile.open(directory / "transformers" / "transformers.tar.gz") as tar_file:
        tar_file.extractall(path=directory / "transformers")
    transformers = model_fn(directory / "transformers")
    transformed_df = predict_fn(df, transformers)[0]
    assert len(transformed_df) == 9  # total 12 - 3 target = 9


def test_predict_fn_except_route(directory):
    input_content = """
        Torgersen,39.2,19.6,195,4675,MALE
    """
    input_content_type = "text/csv"
    df = input_fn(input_content, input_content_type)
    # drop a column arbitrarily so the ColumnTransformer raiser an error about a missing column
    df = df.drop(columns="island")

    with tarfile.open(directory / "transformers" / "transformers.tar.gz") as tar_file:
        tar_file.extractall(path=directory / "transformers")
    transformers = model_fn(directory / "transformers")

    with pytest.raises(Exception):
        predict_fn(df, transformers)


def test_output_fn_try_route(directory):
    input_content = """
        Torgersen,39.2,19.6,195,4675,MALE
    """
    input_content_type = "text/csv"
    df = input_fn(input_content, input_content_type)

    with tarfile.open(directory / "transformers" / "transformers.tar.gz") as tar_file:
        tar_file.extractall(path=directory / "transformers")
    transformers = model_fn(directory / "transformers")
    transformed_input = predict_fn(df, transformers)[0]
    accept = "application/json"
    x, y = output_fn(transformed_input, accept)
    print(transformed_input)
    print(x["instances"])
    for i in range(len(transformed_input)):
        assert x["instances"][i] == transformed_input[i]
    assert y == accept
