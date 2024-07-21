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

    yield base_temp_dir

    shutil.rmtree(base_temp_dir)

def test_train_job_save_a_folder_with_model_artifacts(directory):
    assert "assets" in os.listdir(directory / "model")
    assert "saved_model.pb" in os.listdir(directory / "model" / "assets")