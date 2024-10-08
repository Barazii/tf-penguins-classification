import sys
import os

sys.path.extend([os.path.abspath('.')])
sys.path.extend([os.path.abspath('./code')])

import pytest
import tempfile
from pathlib import Path
import shutil
from constants import *
from processing import process
from training import train
from evaluation import evaluate
import tarfile
import json


@pytest.fixture(scope="function", autouse=False)
def directory(monkeypatch):
    # sagemaker run the processing step and evaluation step in a brand new container.
    # create a temporary base directory for the processing step container
    pc_base_directory = tempfile.mktemp()
    pc_base_directory = Path(pc_base_directory)
    data_dir = pc_base_directory / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_dir / "data.csv")

    process(pc_base_directory)
    
    # temporarily set the env variable SM_MODEL_DIR
    monkeypatch.setenv("SM_MODEL_DIR", f"{pc_base_directory}/model")
    train(
        train_data_dir=pc_base_directory / "transformed-data",
        valid_data_dir=pc_base_directory / "transformed-data",
        hp_epochs=1,
    )

    # create a temporary base directory for the evaluation step container
    ec_base_directory = tempfile.mktemp()
    ec_base_directory = Path(ec_base_directory)
    eval_data_dir = ec_base_directory / "evaluation-data"
    eval_data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pc_base_directory / "baseline", eval_data_dir, dirs_exist_ok=True)

    with tarfile.open(pc_base_directory / "model" / "model.tar.gz", "w:gz") as tar_file:
        tar_file.add(pc_base_directory / "model" / "001", arcname="001")
        tar_file.add(pc_base_directory / "model" / "002", arcname="002")
        tar_file.add(pc_base_directory / "model" / "003", arcname="003")

    eval_model_dir = ec_base_directory / "evaluation-model"
    eval_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pc_base_directory / "model" / "model.tar.gz", eval_model_dir)

    transformers_dir = ec_base_directory / "transformers"
    transformers_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pc_base_directory / "transformers", transformers_dir, dirs_exist_ok=True)

    eval_report_dir = ec_base_directory / "evaluation-report"
    eval_report_dir.mkdir(parents=True, exist_ok=True)

    evaluate(
        pc_base_directory=ec_base_directory,
    )

    yield ec_base_directory

    shutil.rmtree(ec_base_directory)


def test_evaluation_report_in_the_right_format(directory):
    with open(directory / "evaluation-report" / "evaluation_report.json", "r") as eval_report_file:
        eval_report = json.load(eval_report_file)
        
        assert "metrics" in eval_report
        assert "accuracy" in eval_report["metrics"]
        assert "model_1" in eval_report["metrics"]["accuracy"]
        assert "model_2" in eval_report["metrics"]["accuracy"]
        assert "model_3" in eval_report["metrics"]["accuracy"]