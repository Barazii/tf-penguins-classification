import joblib
import pandas as pd
from io import StringIO
import json
from pathlib import Path
import logging

try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError or ModuleNotFoundError:
    worker = None

FEATURE_COLUMNS = [
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
logger = logging.getLogger("preprocessing_component")


def model_fn(transformer_dir):
    logger.info("Function model_fn called.")
    transformer_dir = Path(transformer_dir)
    return joblib.load(transformer_dir / "transformers_1.joblib")

def input_fn(input_content, input_content_type):
    logger.info("Function input_fn called.")
    if input_content == None:
        raise ValueError("No input data received.")
    if input_content_type == "text/csv":
        df = pd.read_csv(StringIO(input_content), header=None, skipinitialspace=True)
        if len(df.columns) == len(FEATURE_COLUMNS) + 1: # for quality check step where data includes the label col
            df = df.drop(columns=df.columns[-1], axis=1)
        df.columns = FEATURE_COLUMNS
        return df

    elif input_content_type == "application/json":
        df = pd.DataFrame([json.loads(input_content)])
        if len(df.columns) == len(FEATURE_COLUMNS) + 1: # for quality check step where data includes the label col
            df = df.drop(columns=df.columns[-1], axis=1)
        df.columns = FEATURE_COLUMNS
        return df

    else: 
        raise ValueError(f"Content type {input_content_type} is not supported.")

def predict_fn(input_data, transformer):
    logger.info("Function predict_fn called.")
    try:
        transformed_input = transformer.transform(input_data)
        return transformed_input
    except:
        raise Exception("Error transforming the input data.")
    
def output_fn(transformed_input, accept):
    logger.info("Function output_fn called.")
    output = {"instances": transformed_input.tolist()}

    try:
        return(
            worker.Response(json.dumps(output), mimetype=accept)
            if worker
            else (output, accept)
        )
    except:
        raise Exception("Failure in sending the transformer output")