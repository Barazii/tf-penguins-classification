from pathlib import Path 
import json
import boto3
from constants import bucket
import joblib
import pandas as pd
import requests
import numpy as np
import io
import tempfile
import tarfile


TARGET_CATEGORIES = ['Adelie', 'Chinstrap', 'Gentoo']

file_obj = io.BytesIO()
boto3.client('s3').download_fileobj(Bucket=bucket, Fileobj=file_obj, Key='processing-step/transformers/transformers.tar.gz')
file_obj.seek(0)
temp_directory = tempfile.mkdtemp()
transformer_directory = Path(temp_directory) / "transformers"
with tarfile.open(fileobj=file_obj, mode="r:gz") as tar:
    tar.extractall(path=transformer_directory) 
transformer_1 = joblib.load(transformer_directory / "transformers_1.joblib")


def handler(data, context, transformers=transformer_1):
    """
    This is the entrypoint that will be called by SageMaker
    when the endpoint receives a request.
    """
    print("Handling endpoint request")

    processed_input = _process_input(data, context, transformer_1)
    output = _predict(processed_input, context) if processed_input else None
    return _process_output(output, context)

def _process_input(data, context, transformers):
    print("Processing input data...")

    if context.request_content_type in (
        "application/json",
        "application/octet-stream",
    ):
        # When the endpoint is running, we will receive a context
        # object. We need to parse the input and turn it into
        # JSON in that case.
        endpoint_input = data.read().decode("utf-8")
    else:
        raise ValueError(
            f"Unsupported content type: {context.request_content_type or 'unknown'}"
        )

    # Let's now transform the input data using the features pipeline.
    try:
        endpoint_input = json.loads(endpoint_input)
        df = pd.json_normalize(endpoint_input)
        result = transformers.transform(df)
    except Exception as e:
        print(f"There was an error processing the input data. {e}")
        return None
    assert result[1] == "application/json"
    return result[0].tolist()

def _predict(instance, context):
    print("Sending input data to model to make a prediction...")

    # When the endpoint is running, we will receive a context
    # object. In that case we need to send the instance to the
    # model to get a prediction back.
    input_data = json.dumps({"instances": [instance]})
    response = requests.post(context.rest_uri, data=input_data)

    if response.status_code != 200:
        raise ValueError(response.content.decode("utf-8"))

    result = json.loads(response.content)
    return result

def _process_output(output, context):
    print("Processing prediction received from the model...")

    if output:
        prediction = np.argmax(output["predictions"][0])
        confidence = output["predictions"][0][prediction]

        result = {
            "prediction": TARGET_CATEGORIES[prediction],
            "confidence": confidence,
        }
    else:
        result = {"prediction": None}

    response_content_type =  context.accept_header
    return json.dumps(result), response_content_type