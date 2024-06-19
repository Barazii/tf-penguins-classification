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
def directory():
    base_temp_dir = tempfile.mktemp()
    data_dir = Path(base_temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_dir / "data.csv")

    base_temp_dir = Path(base_temp_dir)

    process(base_temp_dir)
    train(
        train_data_dir=base_temp_dir / "data-splits",
        model_dir=base_temp_dir / "model",
        hp_batch_size=32,
        hp_epochs=50,
    )

    yield base_temp_dir

    shutil.rmtree(base_temp_dir)

def test_train_job_save_a_folder_with_model_artifacts(directory):
    pass