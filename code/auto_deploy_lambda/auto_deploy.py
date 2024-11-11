from sagemaker.lambda_helper import Lambda
from sagemaker.session import Session
from constants import *
import boto3
import json


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
                                "Service": [
                                    "lambda.amazonaws.com",
                                    "events.amazonaws.com",
                                ]
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
        print(
            f'Role "{lambda_role_name}" already exists with ARN "{response["Role"]["Arn"]}".'
        )
        return response["Role"]["Arn"]


def setup_auto_deploy_lambda():
    # set up lambda function for auto deployment after model approval.
    session = Session()
    lambda_role_arn = create_lambda_role_arn()
    deploy_lambda_fn = Lambda(
        function_name="auto_deploy",
        execution_role_arn=lambda_role_arn,
        script="code/lambda_handler.py",
        handler="lambda_handler.lambda_handler",
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
