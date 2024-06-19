import sys
import os

sys.path.extend([os.path.abspath('.')])
sys.path.extend([os.path.abspath('./code')])

import pytest
import tempfile
from pathlib import Path
import shutil
from constants import *
import subprocess


@pytest.fixture(scope="function", autouse=False)
def directory():
    base_temp_dir = tempfile.mktemp()
    data_dir = Path(base_temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CLEANED_DATA_PATH, data_dir / "data.csv")

    base_temp_dir = Path(base_temp_dir)
    processing_args = ['--pc_base_directory', f"{base_temp_dir}",]
    training_args = ['--epochs', "50",
                     '--batch_size', "32",
                     '--channel_train', f"{base_temp_dir}",
                     '--model_dir', os.path.join(f"{base_temp_dir}", "model"),]

    subprocess.run(['python3', 
                    '/home/mahmood/ml-penguins-classification/code/processing.py'] + 
                    processing_args).check_returncode()
    subprocess.run(['python3', 
                    '/home/mahmood/ml-penguins-classification/code/training.py'] + 
                    training_args).check_returncode()

    yield base_temp_dir

    shutil.rmtree(base_temp_dir)

def test_train_job_save_a_folder_with_model_artifacts(directory):
    pass