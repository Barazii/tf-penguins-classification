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
    base_temp_dir = tempfile.mktemp()
    data_dir = Path(base_temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_dir / "data.csv")

    base_temp_dir = Path(base_temp_dir)

    process(base_temp_dir)
    
    # temporarily set the env variable SM_MODEL_DIR
    monkeypatch.setenv("SM_MODEL_DIR", f"{base_temp_dir}/model")
    train(
        train_data_dir=base_temp_dir / "data-splits",
        hp_epochs=1,
    )

    with tarfile.open(base_temp_dir / "model.tar.gz", "w:gz") as tar_file:
        tar_file.add(base_temp_dir / "model", arcname="trained-model")

    evaluate(
        model_dir=base_temp_dir,
        eval_data_dir=base_temp_dir / "data-splits",
        output_dir=base_temp_dir / "evaluation-report"
    )

    yield base_temp_dir

    shutil.rmtree(base_temp_dir)

def test_evaluation_script_generates_some_output_folders_and_files(directory):
    assert "model.tar.gz" in os.listdir(directory)
    assert "evaluation_report.json" in os.listdir(directory / "evaluation-report")
    assert "trained-model" in os.listdir(directory)

def test_evaluation_report_in_the_right_format(directory):
    with open(directory / "evaluation-report" / "evaluation_report.json", "r") as eval_report_file:
        eval_report = json.load(eval_report_file)
        
        assert "metrics" in eval_report
        assert "accuracy" in eval_report["metrics"]