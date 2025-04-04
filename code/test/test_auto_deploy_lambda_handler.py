import pytest
from auto_deploy_lambda.lambda_handler import lambda_handler
from dotenv import load_dotenv
import boto3
import os


load_dotenv()


@pytest.fixture(scope="function", autouse=False)
def event():
    event = {
        "version": "0",
        "id": "844e2571-85d4-695f-b930-0153b71dcb42",
        "detail-type": "SageMaker Model Package State Change",
        "source": "aws.sagemaker",
        "account": "123456789012",
        "time": "2021-02-24T17:00:14Z",
        "region": "us-east-2",
        "resources": [
            "arn:aws:sagemaker:us-east-2:123456789012:model-package/versionedmp-p-idy6c3e1fiqj/2"
        ],
        "source": ["aws.sagemaker"],
        "detail": {
            "ModelPackageGroupName": "versionedmp-p-idy6c3e1fiqj",
            "ModelPackageVersion": 2,
            "ModelPackageArn": "arn:aws:sagemaker:us-east-2:123456789012:model-package/versionedmp-p-idy6c3e1fiqj/2",
            "CreationTime": "2021-02-24T17:00:14Z",
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": "257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3",
                        "ImageDigest": "sha256:4dc8a7e4a010a19bb9e0a6b063f355393f6e623603361bd8b105f554d4f0c004",
                        "ModelDataUrl": "s3://sagemaker-project-p-idy6c3e1fiqj/versionedmp-p-idy6c3e1fiqj/AbaloneTrain/pipelines-4r83jejmhorv-TrainAbaloneModel-xw869y8C4a/output/model.tar.gz",
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
            "ModelPackageStatus": "Completed",
            "ModelPackageStatusDetails": {
                "ValidationStatuses": [],
                "ImageScanStatuses": [],
            },
            "CertifyForMarketplace": "false",
            "ModelApprovalStatus": "Rejected",
            "MetadataProperties": {
                "GeneratedBy": "arn:aws:sagemaker:us-east-2:123456789012:pipeline/versionedmp-p-idy6c3e1fiqj/execution/4r83jejmhorv"
            },
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": "s3://sagemaker-project-p-idy6c3e1fiqj/versionedmp-p-idy6c3e1fiqj/script-2021-02-24-10-55-15-413/output/evaluation/evaluation.json",
                    }
                }
            },
            "LastModifiedTime": "2021-02-24T17:00:14Z",
        },
    }

    return event


def test_approval_status_rejected(event):
    event["detail"]["ModelApprovalStatus"] = "Rejected"

    lambda_handler(event=event, context=None)


def test_approval_status_approved(event, monkeypatch):
    event["detail"]["ModelApprovalStatus"] = "Approved"
    # mock environment variables
    monkeypatch.setenv("ENDPOINT", os.environ["ENDPOINT"])
    monkeypatch.setenv(
        "DATA_CAPTURE_DESTINATION", os.environ["DATA_CAPTURE_DESTINATION"]
    )
    monkeypatch.setenv("ROLE", os.environ["SM_EXEC_ROLE"])

    # mock the boto3 client calls
    class MockSageMakerClient:
        def create_model(self, ModelName, ExecutionRoleArn, Containers):
            pass

        def create_endpoint_config(
            self, EndpointConfigName, ProductionVariants, DataCaptureConfig
        ):
            pass

        def list_endpoints(self, NameContains, MaxResults):
            return {"Endpoints": [None]}

        def create_endpoint(self, EndpointName, EndpointConfigName):
            pass

        def update_endpoint(self, EndpointName, EndpointConfigName):
            pass

    # Mock boto3.client to return our mock client
    monkeypatch.setattr(boto3, "client", lambda service: MockSageMakerClient())

    lambda_handler(event=event, context=None)
