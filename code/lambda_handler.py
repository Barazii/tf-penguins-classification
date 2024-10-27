import json
import boto3
import time
import os


sagemaker = boto3.client("sagemaker")
instance_type = "ml.m5.xlarge"

def lambda_handler(event, context):
    if "detail" in event:
        model_package_arn = event["detail"]["ModelPackageArn"]
        approval_status = event["detail"]["ModelApprovalStatus"]
    else:
        model_package_arn = event["model_package_arn"]
        approval_status = "Approved"

    # We only want to deploy the approved models
    if approval_status != "Approved":
        response = {
            "message": "Skipping deployment.",
            "approval_status": approval_status,
        }

        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }    
    
    endpoint_name = os.environ["ENDPOINT"]
    data_capture_destination = os.environ["DATA_CAPTURE_DESTINATION"]
    role = os.environ["ROLE"]

    timestamp = time.strftime("%m%d%H%M%S", time.localtime())
    model_name = f"{endpoint_name}-model-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    sagemaker.create_model(
        ModelName=model_name, 
        ExecutionRoleArn=role, 
        Containers=[{
            "ModelPackageName": model_package_arn
        }] 
    )

    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            "ModelName": model_name,
            "InstanceType": f"{instance_type}",
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
            "VariantName": "AllTraffic",
        }],

        # We can enable Data Capture to record the inputs and outputs
        # of the endpoint to use them later for monitoring the model. 
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": data_capture_destination,
            "CaptureOptions": [
                {
                    "CaptureMode": "Input"
                },
                {
                    "CaptureMode": "Output"
                },
            ],
            "CaptureContentTypeHeader": {
                "CsvContentTypes": [
                    "text/csv",
                    "application/octect-stream"
                ],
                "JsonContentTypes": [
                    "application/json",
                    "application/octect-stream"
                ]
            }
        },
    )

    response = sagemaker.list_endpoints(NameContains=endpoint_name, MaxResults=1)

    if len(response["Endpoints"]) == 0:
        # If the endpoint doesn't exist, let's create it.
        sagemaker.create_endpoint(
            EndpointName=endpoint_name, 
            EndpointConfigName=endpoint_config_name,
        )
    else:
        # If the endpoint already exist, let's update it with the
        # new configuration.
        sagemaker.update_endpoint(
            EndpointName=endpoint_name, 
            EndpointConfigName=endpoint_config_name,
        )
    
    return {
        "statusCode": 200,
        "body": json.dumps("Endpoint deployed successfully")
    }
