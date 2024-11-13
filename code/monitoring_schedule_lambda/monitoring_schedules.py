import subprocess
import boto3
import json
import os
import time


def create_lambda_execution_role():

    iam_client = boto3.client("iam")
    lambda_role_name = "monitoring-schedule-role"

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
        )

        iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            RoleName=lambda_role_name,
        )

        iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            RoleName=lambda_role_name,
        )

        iam_client.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
            RoleName=lambda_role_name,
        )

        return response["Role"]["Arn"]

    except iam_client.exceptions.EntityAlreadyExistsException:
        response = iam_client.get_role(RoleName=lambda_role_name)
        print(
            f'Role "{lambda_role_name}" already exists with ARN "{response["Role"]["Arn"]}".'
        )
        return response["Role"]["Arn"]


def setup_monitoring_schedules_lambda():
    # log in
    command = "aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 482497089777.dkr.ecr.eu-north-1.amazonaws.com"
    res = subprocess.run(
        command,
        shell=True,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if res.returncode != 0:
        raise Exception(f"Error in login {res.stderr}")

    # build docker image
    build_cmd = "docker build -t ms-lambda-image ./code/monitoring_schedule_lambda"
    res = subprocess.run(build_cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise Exception(f"Error in building docker image {res.stderr}")

    # tag image
    tag_cmd = "docker tag ms-lambda-image 482497089777.dkr.ecr.eu-north-1.amazonaws.com/lambda-repo"
    res = subprocess.run(tag_cmd, shell=True, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        raise Exception(f"Error in tagging docker image {res.stderr}")

    # create ecr repository to push images to
    ecr_client = boto3.client("ecr")
    response = ecr_client.describe_repositories(
        repositoryNames=[
            "lambda-repo",
        ],
    )
    if not response:  # if no repository found, create one
        create_repo_cmd = "aws ecr create-repository --repository-name lambda-repo"
        res = subprocess.run(
            create_repo_cmd, shell=True, text=True, stderr=subprocess.STDOUT
        )
        if res.returncode != 0:
            raise Exception(f"Error in creating ecr repository {res.stderr}")

    # push image
    push_cmd = "docker push 482497089777.dkr.ecr.eu-north-1.amazonaws.com/lambda-repo"
    res = subprocess.run(push_cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise Exception(f"Error in pushing image to repository {res.stderr}")

    # load script
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        Filename="/home/mahmood/ml-penguins-classification/code/data_quality_monitoring_preprocessing.py",
        Bucket=os.environ["BUCKET"],
        Key="monitoring/data_quality_monitoring_preprocessing.py",
    )

    # create lambda function
    role = create_lambda_execution_role()
    lambda_client = boto3.client("lambda")
    repo_info = ecr_client.describe_repositories(repositoryNames=["lambda-repo"])
    repository_uri = repo_info["repositories"][0]["repositoryUri"]
    image_uri = f"{repository_uri}:latest"
    try:
        lambda_response = lambda_client.create_function(
            FunctionName="ms",
            Role=role,
            Code={"ImageUri": image_uri},
            Description="Data and model monitoring schedules created automatically once the endoint is created and in active state.",
            Timeout=600,
            MemorySize=512,
            PackageType="Image",
            Environment={
                "Variables": {
                    "ENDPOINT": os.environ["ENDPOINT"],
                    "INSTANCE_TYPE": os.environ["INSTANCE_TYPE"],
                    "DATA_QUALITY_LOCATION": os.environ["DATA_QUALITY_LOCATION"],
                    "GROUND_TRUTH_LOCATION": os.environ["GROUND_TRUTH_LOCATION"],
                    "MODEL_QUALITY_LOCATION": os.environ["MODEL_QUALITY_LOCATION"],
                    "MONITORING_PREPROCESSING_SCRIPT": os.environ[
                        "MONITORING_PREPROCESSING_SCRIPT"
                    ],
                    "SM_EXEC_ROLE": os.environ["SM_EXEC_ROLE"],
                }
            },
            Architectures=["x86_64"],
            EphemeralStorage={"Size": 512},
            LoggingConfig={
                "LogFormat": "JSON",
                "ApplicationLogLevel": "TRACE",
                "SystemLogLevel": "DEBUG",
            },
        )
    except lambda_client.exceptions.ResourceConflictException:
        lambda_response = lambda_client.update_function_code(
            FunctionName="ms",
            ImageUri=image_uri,
        )
        time.sleep(60)
        lambda_response = lambda_client.update_function_configuration(
            FunctionName="ms",
            Environment={
                "Variables": {
                    "ENDPOINT": os.environ["ENDPOINT"],
                    "INSTANCE_TYPE": os.environ["INSTANCE_TYPE"],
                    "DATA_QUALITY_LOCATION": os.environ["DATA_QUALITY_LOCATION"],
                    "GROUND_TRUTH_LOCATION": os.environ["GROUND_TRUTH_LOCATION"],
                    "MODEL_QUALITY_LOCATION": os.environ["MODEL_QUALITY_LOCATION"],
                    "MONITORING_PREPROCESSING_SCRIPT": os.environ[
                        "MONITORING_PREPROCESSING_SCRIPT"
                    ],
                    "SM_EXEC_ROLE": os.environ["SM_EXEC_ROLE"],
                }
            },
        )
        print(f'Function {lambda_response["FunctionName"]} already exists. Updated.')

    # event pattern
    event_pattern = """
    {
        "source": [
            "aws.sagemaker"
        ],
        "detail-type": [
            "SageMaker Endpoint State Change"
        ],
        "detail": {
            "EndpointStatus": [
            "IN_SERVICE"
            ],
            "EndpointName": [
            "penguins-endpoint"
            ],
            "EndpointArn": [
            "arn:aws:sagemaker:eu-north-1:482497089777:endpoint/penguins-endpoint"
            ]
        }
    }
    """
    events_client = boto3.client("events")
    rule_response = events_client.put_rule(
        Name="EndpointStateChange",
        EventPattern=event_pattern,
        State="ENABLED",
        RoleArn=role,
    )
    events_client.put_targets(
        Rule="EndpointStateChange",
        Targets=[
            {
                "Id": "1",
                "Arn": lambda_response["FunctionArn"],
            }
        ],
    )
    try:
        response = lambda_client.add_permission(
            Action="lambda:InvokeFunction",
            FunctionName=lambda_response["FunctionName"],
            Principal="events.amazonaws.com",
            SourceArn=rule_response["RuleArn"],
            StatementId="EventBridge",
        )
    except lambda_client.exceptions.ResourceConflictException:
        print(f'Function "{lambda_response["FunctionName"]}" is already invokeable.')


if __name__ == "__main__":
    setup_monitoring_schedules_lambda()
