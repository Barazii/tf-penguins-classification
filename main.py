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
from sagemaker.model_monitor import CronExpressionGenerator, DefaultModelMonitor
from sagemaker.model_monitor import ModelQualityMonitor, EndpointInput


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

    # 2. the model we trained
    tf_model = TensorFlowModel(
        name="trained-model",
        model_data=model_assets,
        framework_version=tf_version,
        sagemaker_session=pipeline_session,
        role=role,
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

    # set up quality monitoring for model and data
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
                    "train-baseline.csv"
                ]
            ),
            dataset_format=DatasetFormat.csv(header=False, output_columns_position="START"),
            output_s3_uri=DATA_QUALITY_LOCATION,
        ),
        model_package_group_name=MODEL_GROUP_NAME,
        skip_check=True,
        register_new_baseline=True,
        cache_config=cache_config,
    )
    create_model_step = ModelStep(
        name="create-model",
        step_args=inference_model.create(instance_type=instance_type),
    )
    transformer = Transformer(
        model_name=create_model_step.properties.ModelName,
        instance_type=instance_type,
        instance_count=1,
        strategy="MultiRecord",
        accept="text/csv",
        assemble_with="Line",
        output_path=f"{s3_project_uri}/transform",
        sagemaker_session=pipeline_session,
    )
    generate_test_predictions_step = TransformStep(
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
            # The first field corresponds to the groundtruth,
            # and the second to last field corresponds to the transform output.
            #
            # Here is an example of the data generated
            # after joining the input with the transform output:
            #
            # Gentoo,39.1,18.7,181.0,3750.0,MALE,Gentoo,0.52
            #
            # Notice how the first field is the groundtruth coming from the
            # test set. The second to last field is the prediction coming the
            # model.
            # output_filter="$[0,-2]",
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
            baseline_dataset=generate_test_predictions_step.properties.TransformOutput.S3OutputPath,
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
        model_package_group_name=MODEL_GROUP_NAME,
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
            model_package_group_name=MODEL_GROUP_NAME,
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
    
    # set up the lambda function
    lambda_role_arn = create_lambda_role_arn()

    deploy_lambda_fn = Lambda(
        function_name="deploy_fn",
        execution_role_arn=lambda_role_arn,
        script="code/lambda.py",
        handler="lambda.lambda_handler",
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

    # set up the event bridge for lambda step 
    event_pattern = f"""
    {{
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Model Package State Change"],
    "detail": {{
        "ModelPackageGroupName": ["{MODEL_GROUP_NAME}"],
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

    # set up the condition step
    accuracy_threshold = ParameterFloat(name="accuracy_threshold", default_value=0.65)
    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=eval_step.name,
            property_file=eval_report,
            json_path="metrics.accuracy",
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
            create_model_step, 
            generate_test_predictions_step,
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

    # start the pipeline
    ret = pipeline.start()

    # monitoring jobs
    print("waiting for pipeline execution...")
    ret.wait(delay=180)
    print("execution finished")
    # s3_client.upload_file(Filename="./code/data_quality_preprocessing.py", Bucket=bucket, Key=MONITORING_SCRIPT_LOCATION)

    # sagemaker_client = boto3.client("sagemaker")

    # # data quality monitoring job
    # try:
    #     data_monitor = DefaultModelMonitor(
    #         instance_type=instance_type,
    #         instance_count=1,
    #         max_runtime_in_seconds=3600,
    #         role=role,
    #     )
    #     data_monitor.create_monitoring_schedule(
    #         monitor_schedule_name="data-monitoring-schedule",
    #         endpoint_input=ENDPOINT,
    #         record_preprocessor_script=data_quality_preprocessing_uri,
    #         statistics=f"{DATA_QUALITY_LOCATION}/statistics.json",
    #         constraints=f"{DATA_QUALITY_LOCATION}/constraints.json",
    #         schedule_cron_expression=CronExpressionGenerator.hourly(),
    #         output_s3_uri=DATA_QUALITY_LOCATION,
    #         enable_cloudwatch_metrics=True,
    #     )
    # except sagemaker_client.exceptions.ResourceInUse as _:
    #     boto3.client("sagemaker").delete_monitoring_schedule(MonitoringScheduleName='data-monitoring-schedule')
    #     import time
    #     time.sleep(10)
    #     data_monitor = DefaultModelMonitor(
    #         instance_type=instance_type,
    #         instance_count=1,
    #         max_runtime_in_seconds=3600,
    #         role=role,
    #     )
    #     data_monitor.create_monitoring_schedule(
    #         monitor_schedule_name="data-monitoring-schedule",
    #         endpoint_input=ENDPOINT,
    #         record_preprocessor_script=data_quality_preprocessing_uri,
    #         statistics=f"{DATA_QUALITY_LOCATION}/statistics.json",
    #         constraints=f"{DATA_QUALITY_LOCATION}/constraints.json",
    #         schedule_cron_expression=CronExpressionGenerator.hourly(),
    #         output_s3_uri=DATA_QUALITY_LOCATION,
    #         enable_cloudwatch_metrics=True,
    #     )
    
    # # model quality monitoring job
    # try:
    #     model_monitor = ModelQualityMonitor(
    #         instance_type=instance_type,
    #         instance_count=1,
    #         max_runtime_in_seconds=1800,
    #         role=role
    #     )
    #     model_monitor.create_monitoring_schedule(
    #         monitor_schedule_name="model-monitoring-schedule",
    #         endpoint_input = EndpointInput(
    #             endpoint_name=ENDPOINT,
    #             inference_attribute="prediction",
    #             destination="/opt/ml/processing/input_data",
    #         ),
    #         problem_type="MulticlassClassification",
    #         ground_truth_input=GROUND_TRUTH_LOCATION,
    #         constraints=f"{MODEL_QUALITY_LOCATION}/constraints.json",
    #         schedule_cron_expression=CronExpressionGenerator.hourly(),
    #         output_s3_uri=MODEL_QUALITY_LOCATION,
    #         enable_cloudwatch_metrics=True,
    #     )
    # except sagemaker_client.exceptions.ResourceInUse as _:
    #     boto3.client("sagemaker").delete_monitoring_schedule(MonitoringScheduleName='model-monitoring-schedule')
    #     import time
    #     time.sleep(10)
    #     model_monitor = ModelQualityMonitor(
    #         instance_type=instance_type,
    #         instance_count=1,
    #         max_runtime_in_seconds=1800,
    #         role=role
    #     )
    #     model_monitor.create_monitoring_schedule(
    #         monitor_schedule_name="model-monitoring-schedule",
    #         endpoint_input = EndpointInput(
    #             endpoint_name=ENDPOINT,
    #             inference_attribute="prediction",
    #             destination="/opt/ml/processing/input_data",
    #         ),
    #         problem_type="MulticlassClassification",
    #         ground_truth_input=GROUND_TRUTH_LOCATION,
    #         constraints=f"{MODEL_QUALITY_LOCATION}/constraints.json",
    #         schedule_cron_expression=CronExpressionGenerator.hourly(),
    #         output_s3_uri=MODEL_QUALITY_LOCATION,
    #         enable_cloudwatch_metrics=True,
    #     )
    # # describe_data_monitoring_schedule(ENDPOINT)