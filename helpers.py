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