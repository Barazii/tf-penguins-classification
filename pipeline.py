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
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.pipeline import PipelineModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.fail_step import FailStep
from helpers import *
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.quality_check_step import QualityCheckStep, DataQualityCheckConfig, ModelQualityCheckConfig
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.drift_check_baselines import DriftCheckBaselines
import threading
from code.lambda_setup import set_up_lambda_fn


if __name__ == "__main__":

    pipeline_session = PipelineSession(default_bucket=bucket)
    session = Session()
    s3_client = boto3.client("s3")
    cache_config = CacheConfig(enable_caching=False, expire_after="5d")

    # transfer data to s3
    s3_client.upload_file(Filename=CLEANED_DATA_PATH, Bucket=bucket, Key="data/data.csv")

    # set up the processing step 
    processor = SKLearnProcessor(
        base_job_name="data-processing-processor",
        framework_version=skl_version,
        instance_type=instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        tags={"Key": "tagkey", "Value":"tagvalue"},
    )
    processing_step = ProcessingStep(
        name="processing-step",
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
                    source=os.path.join(pc_base_directory, "transformed-data"),
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
                "validation": TrainingInput(
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
            dependencies=["requirements.txt"],
            code=f"./code/evaluation.py",
            arguments=[
                "--pc_base_directory", f"{pc_base_directory}",
            ],
            inputs=[
                ProcessingInput(
                    input_name="evaluation-data",
                    source=processing_step.properties.ProcessingOutputConfig.Outputs["baseline"].S3Output.S3Uri,
                    destination=f"{pc_base_directory}/evaluation-data",
                ),
                ProcessingInput(
                    input_name="transformers",
                    source=processing_step.properties.ProcessingOutputConfig.Outputs["transformers"].S3Output.S3Uri,
                    destination=f"{pc_base_directory}/transformers",
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

    # build the inference pipeline (preprocessing model, trained model, postprocessing model)
    # 1. the pre processing model
    transformers_uri = Join(
        on="/",
        values=[
            processing_step.properties.ProcessingOutputConfig.Outputs["transformers"].S3Output.S3Uri,
            "transformers.tar.gz"
        ]
    )
    preprocessing_model = SKLearnModel(
        name="preprocessing-model",
        model_data=transformers_uri,
        entry_point=f"./code/preprocessing_component.py",
        framework_version=skl_version,
        sagemaker_session=pipeline_session,
        role=role,
    )

    env = {
        'SAGEMAKER_TFS_DEFAULT_MODEL_NAME': 'model1'
    }
    # 2. the model we trained
    tf_model = TensorFlowModel(
        name="trained-model",
        model_data=model_assets,
        framework_version=tf_version,
        sagemaker_session=pipeline_session,
        role=role,
        env=env,
    )

    # 3. the post processing model
    postprocessing_model = SKLearnModel(
        name="postprocessing-model",
        model_data=transformers_uri,
        entry_point=f"./code/postprocessing_component.py",
        framework_version=skl_version,
        sagemaker_session=pipeline_session,
        role=role,
    )

    # build the inference pipeline
    inference_model = PipelineModel(
        name="inference-model",
        models=[preprocessing_model, tf_model, postprocessing_model],
        sagemaker_session=pipeline_session,
        role=role,
    )

    # set up quality baselines for model and data
    # Data quality baseline
    data_quality_baseline_step = QualityCheckStep(
        name="generate-data-quality-baseline",
        check_job_config=CheckJobConfig(
            instance_type="ml.c5.xlarge",
            instance_count=1,
            volume_size_in_gb=20,
            sagemaker_session=pipeline_session,
            role=role,
        ),
        quality_check_config=DataQualityCheckConfig(
            baseline_dataset=Join(
                on="/",
                values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs["baseline"].S3Output.S3Uri,
                    "train-baseline-1.csv"
                ]
            ),
            dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
            output_s3_uri=DATA_QUALITY_LOCATION,
        ),
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        skip_check=True,
        register_new_baseline=True,
        cache_config=cache_config,
    )
    # Model quality baseline
    model_step = ModelStep(
        name="create-model",
        step_args=inference_model.create(instance_type=instance_type),
    )
    transformer = Transformer(
        model_name=model_step.properties.ModelName,
        instance_type=instance_type,
        instance_count=1,
        strategy="MultiRecord",
        accept="text/csv",
        assemble_with="Line",
        output_path=TRANSFORM_LOCATION,
        sagemaker_session=pipeline_session,
    )
    transform_step = TransformStep(
        name="generate-test-predictions",
        step_args=transformer.transform(
            data=Join(
                on="/",
                values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs["baseline"].S3Output.S3Uri,
                    "test-baseline.csv"
                ]
            ),
            join_source="Input",
            split_type="Line",
            content_type="text/csv",
            # The first field corresponds to the groundtruth coming from the
            # test set, and the second to last field corresponds to the 
            # transform output or the model prediction.
            #
            # Here is an example of the data generated
            # after joining the input with the transform output:
            # Gentoo,Biscoe,39.1,18.7,181.0,3750.0,MALE,Gentoo,0.52
            #
            # We only take groundtruth, predicted class and confidence score
            # to use it for comparison and quality check.
            output_filter="$[-3,-2,-1]",
        ),
        cache_config=cache_config,
    )
    model_quality_baseline_step = QualityCheckStep(
        name="generate-model-quality-baseline",
        check_job_config=CheckJobConfig(
            instance_type="ml.c5.xlarge",
            instance_count=1,
            volume_size_in_gb=20,
            sagemaker_session=pipeline_session,
            role=role,
        ),
        quality_check_config=ModelQualityCheckConfig(
            # We are going to use the output of the Transform Step to generate
            # the model quality baseline.
            baseline_dataset=transform_step.properties.TransformOutput.S3OutputPath,
            dataset_format=DatasetFormat.csv(header=False),

            # We need to specify the problem type and the fields where the prediction
            # and groundtruth are so the process knows how to interpret the results.
            problem_type="MulticlassClassification",
            
            # Since the data doesn't have headers, SageMaker will autocreate headers for it.
            # _c0 corresponds to the first column, and _c1 corresponds to the second column.
            ground_truth_attribute="_c0",
            inference_attribute="_c1",
            output_s3_uri=MODEL_QUALITY_LOCATION,
        ),
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
        skip_check=True,
        register_new_baseline=True,
        cache_config=cache_config,
    )
    model_metrics = ModelMetrics(
        model_data_statistics=MetricsSource(
            s3_uri=data_quality_baseline_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_baseline_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            s3_uri=model_quality_baseline_step.properties.CalculatedBaselineStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_baseline_step.properties.CalculatedBaselineConstraints,
            content_type="application/json",
        ),
    )

    drift_check_baselines = DriftCheckBaselines(
        model_data_statistics=MetricsSource(
            s3_uri=data_quality_baseline_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_data_constraints=MetricsSource(
            s3_uri=data_quality_baseline_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
        model_statistics=MetricsSource(
            s3_uri=model_quality_baseline_step.properties.BaselineUsedForDriftCheckStatistics,
            content_type="application/json",
        ),
        model_constraints=MetricsSource(
            s3_uri=model_quality_baseline_step.properties.BaselineUsedForDriftCheckConstraints,
            content_type="application/json",
        ),
    )

    # set up the registration step
    registration_step = ModelStep(
        name="registration-step",
        display_name="registration-step",
        step_args=inference_model.register(
            model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
            model_metrics=model_metrics,
            drift_check_baselines=drift_check_baselines,
            approval_status="PendingManualApproval",
            content_types=["text/csv", "application/json"],
            response_types=["text/csv", "application/json"],
            inference_instances=[instance_type],
            transform_instances=[instance_type],
            domain="MACHINE_LEARNING",
            task="CLASSIFICATION",
            framework="TENSORFLOW",
            framework_version=tf_version,
            description="commit message here",
        ),
    )

    # set up the condition step
    accuracy_threshold = ParameterFloat(name="accuracy_threshold", default_value=0.6)
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=eval_report,
            json_path="metrics.accuracy.model_1",
        ),
        right=accuracy_threshold
    )
    fail_step = FailStep(
        name="fail-step",
        error_message=Join(
            on=" ",
            values=[
                "Model's accuracy is less than accuracy threshold",
                accuracy_threshold
            ]
        )
    )
    condition_step = ConditionStep(
        name="condition-step",
        conditions=[condition],
        if_steps=[
            model_step, 
            transform_step,
            model_quality_baseline_step,
            registration_step
        ],
        else_steps=[fail_step]
    )

    # build the final pipeline 
    pl_def_config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    pipeline = Pipeline(
        name="penguins-classification-pipeline",
        parameters=[accuracy_threshold],
        steps=[
            processing_step,
            training_step,
            eval_step,
            data_quality_baseline_step,
            condition_step
        ],
        sagemaker_session=pipeline_session,
        pipeline_definition_config=pl_def_config,
    )
    pipeline.upsert(role_arn=role)

    # before starting the pipeline, set up the lambda function in the background 
    # in another thread.
    thread = threading.Thread(target=set_up_lambda_fn)
    thread.start()

    # start the pipeline
    try:
        ret = pipeline.start()
        print("waiting for pipeline execution...")
        ret.wait(delay=180)
        print("execution finished")
    except Exception as e:
        print("Error in running the pipeline.", e)