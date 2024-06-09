from dotenv import load_dotenv
import os 
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.session import Session
import boto3
from sagemaker.workflow.steps import CacheConfig
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep


if __name__ == "__main__":

    CLEANED_DATA_PATH = "./data/penguins_cleaned.csv"
    load_dotenv()
    role = os.environ["ROLE"]
    bucket = os.environ["BUCKET"]
    s3_project_uri = f"s3://{bucket}"
    pipeline_session = PipelineSession()
    pure_session = Session()
    sagemaker_client = boto3.client("sagemaker")
    iam_client = boto3.client("iam")
    s3_client = boto3.client("s3")
    instance_type = "ml.m5.xlarge"
    tf_version = "2.11"
    skl_version = "1.2-1"
    py_version = "py39"
    cache_config = CacheConfig(enable_caching=True, expire_after="5d")

    # transfer data to s3
    s3_client.upload_file(Filename=CLEANED_DATA_PATH, Bucket=bucket, Key="data/data.csv")

    # set up the processing step 
    processor = SKLearnProcessor(
        base_job_name="preprocess-data",
        framework_version=skl_version,
        instance_type=instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        tags={"Key": "tagkey", "Value":"tagvalue"}
    )
    base_directory = "/opt/ml/preprocessing"
    processing_step = ProcessingStep(
        name="preprocess-data",
        cache_config=cache_config,
        step_args=processor.run(
            code="./code/preprocessing.py",
            inputs=[
                ProcessingInput(
                    input_name="input",
                    source=os.path.join(s3_project_uri, "data"),
                    destination=os.path.join(base_directory, "data")
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="data-splits",
                    source=os.path.join(base_directory, "data-splits"),
                    destination=os.path.join(s3_project_uri, "preprocessing/data-splits")
                ),
                ProcessingOutput(
                    output_name="model",
                    source=os.path.join(base_directory, "model"),
                    destination=os.path.join(s3_project_uri, "preprocessing/model")
                ),
                ProcessingOutput(
                    output_name="transformers",
                    source=os.path.join(base_directory, "transformers"),
                    destination=os.path.join(s3_project_uri, "preprocessing/transformer")
                ),
                ProcessingOutput(
                    output_name="baseline",
                    source=os.path.join(base_directory, "baseline"),
                    destination=os.path.join(s3_project_uri, "preprocessing/baseline")
                ),
            ],
        )
    )

    # set up training step 
