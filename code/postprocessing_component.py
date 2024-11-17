import json
import numpy as np
try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError:
    worker = None
import logging

TARGET_CATEGORIES = ['Adelie', 'Chinstrap', 'Gentoo']
logger = logging.getLogger("postprocessing_component")


def model_fn(arg=None):
    logger.info("Function model_fn called.")
    target_categories = TARGET_CATEGORIES
    return target_categories

def input_fn(model_output_probabilities, content_type):
    logger.info("Function input_fn called.")
    if model_output_probabilities == None or len(model_output_probabilities) == 0:
        raise ValueError("No predictions/probabilities received from the model.")
    if content_type == "application/json":
        probabilities = json.loads(model_output_probabilities)["predictions"]
        return probabilities
    
    raise ValueError(f"{content_type} is not supported.")

def predict_fn(probabilities, target_categories):
    logger.info("Function predict_fn called.")
    target_predictions = np.argmax(probabilities, axis=-1)
    confidence = np.max(probabilities, axis=-1)
    return [
        (target_categories[prediction], confidence)
        for confidence, prediction in zip(confidence, target_predictions)
    ]

def output_fn(predictions, accept):
    logger.info("Function output_fn called.")
    if accept == "text/csv":
        return (
            worker.Response(encoders.encode(predictions, accept), mimetype=accept)
            if worker
            else (predictions, accept)
        )

    elif accept == "application/json":
        response = []
        for p, c in predictions:
            response.append({"0": p, "1": c})

        # If there's only one prediction, we'll return it
        # as a single object.
        if len(response) == 1:
            response = response[0]

        return (
            worker.Response(json.dumps(response), mimetype=accept)
            if worker
            else (response, accept)
        )

    else:
        raise ValueError(f"{accept} accept type is not supported.")