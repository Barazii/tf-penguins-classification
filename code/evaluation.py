import argparse
import pandas as pd
from pathlib import Path
import tarfile
import keras
import numpy as np
from sklearn.metrics import accuracy_score
import json


def evaluate(pc_base_directory):
    eval_data_dir = pc_base_directory / "evaluation-data"
    X_test = pd.read_csv(eval_data_dir / "test" / "X_test.csv")
    y_test = pd.read_csv(eval_data_dir / "test" / "y_test.csv")

    model_dir = pc_base_directory / "evaluation-model"
    with tarfile.open(model_dir / "model.tar.gz") as tar_file:
        tar_file.extractall(path=model_dir)

    model = keras.models.load_model(model_dir / "assets")

    predictions = np.argmax(model.predict(np.array(X_test)), axis=-1)
    true_labels = np.argmax(np.array(y_test), axis=-1)
    accuracy = accuracy_score(true_labels, predictions)

    evaluation_report = {
        "metrics": {
            "accuracy": accuracy
        }
    }

    eval_report_dir = pc_base_directory / "evaluation-report"
    eval_report_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_report_dir / "evaluation_report.json", "w") as eval_report_file:
        eval_report_file.write(json.dumps(evaluation_report))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--pc_base_directory", type=str, required=True)
    args = arg_parser.parse_args()
    evaluate(Path(args.pc_base_directory),)