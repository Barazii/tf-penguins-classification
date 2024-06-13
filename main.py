import os 
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.session import Session
import boto3
from sagemaker.workflow.steps import CacheConfig
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from constants import *

if __name__ == "__main__":

    pipeline_session = PipelineSession()
    pure_session = Session()
    sagemaker_client = boto3.client("sagemaker")
    iam_client = boto3.client("iam")
    s3_client = boto3.client("s3")
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
    processing_step = ProcessingStep(
        name="preprocess-data",
        cache_config=cache_config,
        step_args=processor.run(
            code="./code/processing.py",
            arguments=[
                "--pc_base_directory", f"{pc_base_directory}",
            ],
            inputs=[
                ProcessingInput(
                    input_name="data",
                    source=os.path.join(s3_project_uri, "data"),
                    destination=os.path.join(pc_base_directory, "data")
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="data-splits",
                    source=os.path.join(pc_base_directory, "data-splits"),
                    destination=os.path.join(s3_project_uri, "processing-step/data-splits")
                ),
                ProcessingOutput(
                    output_name="model",
                    source=os.path.join(pc_base_directory, "model"),
                    destination=os.path.join(s3_project_uri, "training-step/model")
                ),
                ProcessingOutput(
                    output_name="transformers",
                    source=os.path.join(pc_base_directory, "transformers"),
                    destination=os.path.join(s3_project_uri, "processing-step/transformers")
                ),
                ProcessingOutput(
                    output_name="baseline",
                    source=os.path.join(pc_base_directory, "baseline"),
                    destination=os.path.join(s3_project_uri, "processing-step/baseline")
                ),
            ],
        )
    )

    # set up training step 

    # build the pipeline 
    pl_def_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    pipeline = Pipeline(
        name="penguins-classification",
        steps=[processing_step],
        sagemaker_session=pipeline_session,
        pipeline_definition_config=pl_def_config,
    )
    pipeline.upsert(role_arn=role)

    # start the pipeline
    # pipeline.start()
