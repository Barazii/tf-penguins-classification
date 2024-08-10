import json

def preprocess_handler(inference_record):
    input_data = json.loads(inference_record.endpoint_input.data)
    output_data = json.loads(inference_record.endpoint_output.data)
    
    response = input_data
    response["species"] = output_data["prediction"]

    return response