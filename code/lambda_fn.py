from sagemaker.lambda_helper import Lambda
from sagemaker.session import Session
from constants import *
import boto3
import json
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


def create_lambda_role_arn():

    iam_client = boto3.client("iam")
    lambda_role_name = "lambda-deployment-role"

    try:
        response = iam_client.create_role(
            RoleName=lambda_role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": ["lambda.amazonaws.com", "events.amazonaws.com"]
                            },
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Lambda Endpoint Deployment",
        )

        iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            RoleName=lambda_role_name,
        )

        iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            RoleName=lambda_role_name,
        )

        return response["Role"]["Arn"]

    except iam_client.exceptions.EntityAlreadyExistsException:
        response = iam_client.get_role(RoleName=lambda_role_name)
        print(f'Role "{lambda_role_name}" already exists with ARN "{response["Role"]["Arn"]}".')
        return response["Role"]["Arn"]


def set_up_lambda_fn():
    # set up the lambda function
    session = Session()
    lambda_role_arn = create_lambda_role_arn()
    deploy_lambda_fn = Lambda(
        function_name="deploy_fn",
        execution_role_arn=lambda_role_arn,
        script="code/lambda_fn.py",
        handler="lambda_fn.lambda_handler",
        timeout=600,
        session=session,
        runtime="python3.11",
        environment={
            "Variables": {
                "ENDPOINT": ENDPOINT,
                "DATA_CAPTURE_DESTINATION": DATA_CAPTURE_DESTINATION,
                "ROLE": role,
            }
        },
    )
    lambda_response = deploy_lambda_fn.upsert()

    # set up the event bridge of the lambda function
    event_pattern = f"""
    {{
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Model Package State Change"],
    "detail": {{
        "ModelPackageGroupName": ["{MODEL_PACKAGE_GROUP_NAME}"],
        "ModelApprovalStatus": ["Approved"]
    }}
    }}
    """
    events_client = boto3.client("events")
    rule_response = events_client.put_rule(
        Name="PipelineModelApprovedRule",
        EventPattern=event_pattern,
        State="ENABLED",
        RoleArn=role,
    )
    events_client.put_targets(
        Rule="PipelineModelApprovedRule",
        Targets=[
            {
                "Id": "1",
                "Arn": lambda_response["FunctionArn"],
            }
        ],
    )
    lambda_client = boto3.client("lambda")
    try:
        response = lambda_client.add_permission(
            Action="lambda:InvokeFunction",
            FunctionName=lambda_response["FunctionName"],
            Principal="events.amazonaws.com",
            SourceArn=rule_response["RuleArn"],
            StatementId="EventBridge",
        )
    except lambda_client.exceptions.ResourceConflictException as e:
        print(f'Function "{lambda_response["FunctionName"]}" already has permissions.')