import os 
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.session import Session
import boto3
from sagemaker.workflow.steps import CacheConfig
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from constants import *
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow import TensorFlowProcessor
from sagemaker.workflow.properties import PropertyFile


if __name__ == "__main__":

    pipeline_session = PipelineSession(default_bucket=bucket)
    pure_session = Session()
    sagemaker_client = boto3.client("sagemaker")
    iam_client = boto3.client("iam")
    s3_client = boto3.client("s3")
    cache_config = CacheConfig(enable_caching=False, expire_after="5d")

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
        tags={"Key": "tagkey", "Value":"tagvalue"},
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
    tf_estimator = TensorFlow(
        base_job_name="train-model",
        entry_point=f"./code/training.py",
        hyperparameters={
            "epochs": "20",
            "batch_size": "32",
        },
        metric_definitions=[
            {"Name": "loss", "Regex": "loss: ([0-9\\.]+)"},
            {"Name": "accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
        ],
        framework_version=tf_version,
        py_version=py_version,
        instance_type=instance_type,
        instance_count=1,
        disable_profiler=True,
        sagemaker_session=pipeline_session,
        role=role,
        enable_sagemaker_metrics=True,
    )
    training_step = TrainingStep(
        name="training-step",
        step_args=tf_estimator.fit(
            inputs={
                "train": TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                        "data-splits"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                ),
            },
        ),
        cache_config=cache_config,
    )

    # Evaluation step
    eval_processor = TensorFlowProcessor(
        base_job_name="evaluation-processor",
        framework_version=tf_version,
        py_version=py_version,
        instance_type=instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )
    eval_report = PropertyFile(
        name="evaluation-report",
        output_name="evaluation-report",
        path="evaluation_report.json",
    )
    model_assets = training_step.properties.ModelArtifacts.S3ModelArtifacts
    eval_step = ProcessingStep(
        name="evaluation-step",
        step_args=eval_processor.run(
            code=f"./code/evaluation.py",
            arguments=[
                "--pc_base_directory", f"{pc_base_directory}",
            ],
            inputs=[
                ProcessingInput(
                    input_name="evaluation-data",
                    source=processing_step.properties.ProcessingOutputConfig.Outputs["data-splits"].S3Output.S3Uri,
                    destination=f"{pc_base_directory}/evaluation-data",
                ),
                ProcessingInput(
                    input_name="evaluation-model",
                    source=model_assets,
                    destination=f"{pc_base_directory}/evaluation-model"
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation-report",
                    source=f"{pc_base_directory}/evaluation-report",
                    destination=os.path.join(s3_project_uri, "evaluation-step"),
                ),
            ]
        ),
        property_files=[eval_report],
        cache_config=cache_config,
    )


    # build the pipeline 
    pl_def_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    pipeline = Pipeline(
        name="penguins-classification-pipeline",
        steps=[
            processing_step,
            training_step,
            eval_step,
            
        ],
        sagemaker_session=pipeline_session,
        pipeline_definition_config=pl_def_config,
    )
    pipeline.upsert(role_arn=role)

    # start the pipeline
    # pipeline.start()