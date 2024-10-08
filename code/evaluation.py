import argparse
import pandas as pd
from pathlib import Path
import tarfile
import keras
import numpy as np
from sklearn.metrics import accuracy_score
import json
import joblib
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder


def evaluate(pc_base_directory):
    transformers_dir = pc_base_directory / "transformers"
    with tarfile.open(transformers_dir / "transformers.tar.gz") as tar_file:
        tar_file.extractall(path=transformers_dir)
    transformer_1 = joblib.load(transformers_dir / "transformers_1.joblib")
    transformer_2 = joblib.load(transformers_dir / "transformers_2.joblib")
    transformer_3 = joblib.load(transformers_dir / "transformers_3.joblib")

    eval_data_dir = pc_base_directory / "evaluation-data"
    eval_data = pd.read_csv(eval_data_dir / "test-baseline.csv")
    X_eval_1 = transformer_1.transform(eval_data.drop(columns="species", axis=1))
    X_eval_2 = transformer_2.transform(eval_data.drop(columns="species", axis=1))
    X_eval_3 = transformer_3.transform(eval_data.drop(columns="species", axis=1))

    model_dir = pc_base_directory / "evaluation-model"
    with tarfile.open(model_dir / "model.tar.gz") as tar_file:
        tar_file.extractall(path=model_dir)
    model_1 = keras.models.load_model(model_dir / "001")
    model_2 = keras.models.load_model(model_dir / "002")
    model_3 = keras.models.load_model(model_dir / "003")

    target_transformer = ColumnTransformer(
        transformers=[
            ("categorical_tf", OneHotEncoder(), make_column_selector(dtype_include=object))
        ],
    )
    y_eval = target_transformer.fit_transform(pd.DataFrame(eval_data.species))
    true_labels = np.argmax(np.array(y_eval), axis=-1)

    predictions_1 = np.argmax(model_1.predict(np.array(X_eval_1)), axis=-1)
    accuracy_1 = accuracy_score(true_labels, predictions_1)

    predictions_2 = np.argmax(model_2.predict(np.array(X_eval_2)), axis=-1)
    accuracy_2 = accuracy_score(true_labels, predictions_2)

    predictions_3 = np.argmax(model_3.predict(np.array(X_eval_3)), axis=-1)
    accuracy_3 = accuracy_score(true_labels, predictions_3)

    evaluation_report = {
        "metrics": {
            "accuracy": {
                "model_1": accuracy_1,
                "model_2": accuracy_2,
                "model_3": accuracy_3
            }

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