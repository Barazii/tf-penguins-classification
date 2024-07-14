import sys
import os

sys.path.extend([os.path.abspath('.')])

import joblib
import os
import pandas as pd
from io import StringIO
import json
from constants import FEATURE_COLUMNS
import tarfile

try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError or ModuleNotFoundError:
    worker = None


def model_fn(transformer_dir):
    with tarfile.open(transformer_dir / "transformers.tar.gz") as trans_tar_file:
        trans_tar_file.extractall(path=transformer_dir / "unzipped_transformers_dir")

    return joblib.load(transformer_dir / "unzipped_transformers_dir" / "transformers.joblib")

def input_fn(input_content, input_content_type):

    if input_content_type == "text/csv":
        df = pd.read_csv(StringIO(input_content), header=None, skipinitialspace=True)
        df.columns = FEATURE_COLUMNS
        return df

    elif input_content_type == "application/json":
        df = pd.DataFrame([json.loads(input_content)])
        df.columns = FEATURE_COLUMNS
        return df

    else: 
        raise ValueError(f"Content type {input_content_type} is not supported.")

def predict_fn(input_data, transformer):
    try:
        transformed_input = transformer.transform(input_data)
        return transformed_input
    except ValueError as e:
        print("Error transforming the input data", e)
        return None
    
def output_fn(transformed_input, accept):
    if transformed_input is None:
        raise Exception("There was an error transforming the input data")
    
    output = {"transformed input": transformed_input.tolist()}

    try:
        return(
            worker.Response(json.dumps(output), mimetype=accept)
            if worker
            else (output, accept)
        )
    except:
        raise Exception("Failure in sending the transformer output")