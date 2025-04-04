import os
from dotenv import load_dotenv
import pytest
import tempfile
from pathlib import Path
import shutil
from processing import process
from training import train


load_dotenv()


@pytest.fixture(scope="function", autouse=False)
def directory(monkeypatch):
    base_temp_dir = tempfile.mktemp()
    data_dir = Path(base_temp_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.environ["CLEANED_DATA_PATH"], data_dir / "data.csv")

    base_temp_dir = Path(base_temp_dir)

    process(base_temp_dir)

    # temporarily set the env variable SM_MODEL_DIR
    monkeypatch.setenv("SM_MODEL_DIR", f"{base_temp_dir}/model")
    train(
        train_data_dir=base_temp_dir / "transformed-data",
        valid_data_dir=base_temp_dir / "transformed-data",
        hp_epochs=1,
    )

    yield base_temp_dir

    shutil.rmtree(base_temp_dir)


def test_train_job_save_3_models_artifacts(directory):
    assert "model1" in os.listdir(directory / "model")
    assert "model2" in os.listdir(directory / "model")
    assert "model3" in os.listdir(directory / "model")

    assert "saved_model.pb" in os.listdir(directory / "model" / "model1" / "001")
    assert "saved_model.pb" in os.listdir(directory / "model" / "model2" / "002")
    assert "saved_model.pb" in os.listdir(directory / "model" / "model3" / "003")
