def preprocess_handler(inference_record, logger):
    input_data = inference_record.endpoint_input.data
    logger.info(f"This is data input {input_data}")
    return {str(i).zfill(2): d for i, d in enumerate(input_data.split(","))}