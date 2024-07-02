import argparse
import pandas as pd
from pathlib import Path
import tarfile
import keras
import numpy as np
from sklearn.metrics import accuracy_score
import json


def evaluate(model_dir, eval_data_dir, output_dir):
    X_test = pd.read_csv(Path(eval_data_dir) / "test" / "X_test.csv")
    y_test = pd.read_csv(Path(eval_data_dir) / "test" / "y_test.csv")

    with tarfile.open(Path(model_dir) / "model.tar.gz") as tar_file:
        tar_file.extractall(path=Path(model_dir))

    model = keras.models.load_model(Path(model_dir) / "trained-model")

    predictions = np.argmax(model.predict(X_test), axis=-1)
    true_labels = np.argmax(y_test, axis=-1)
    accuracy = accuracy_score(true_labels, predictions)

    evaluation_report = {
        "metrics": {
            "accuracy": accuracy
        }
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "evaluation_report.json", "w") as eval_report_file:
        eval_report_file.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--pc_base_directory", type=str, required=True)
    args = arg_parser.parse_args()
    evaluate(
        model_dir=f"{args.pc_base_directory}/model",
        eval_data_dir=f"{args.pc_base_directory}/evaluation-data",
        output_dir=f"{args.pc_base_directory}/evaluation-report",
    )