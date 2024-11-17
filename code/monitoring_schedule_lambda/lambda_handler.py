from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DefaultModelMonitor,
    EndpointInput,
)
from sagemaker.model_monitor import ModelQualityMonitor
import botocore
import time
import threading
import boto3
import os
import time
import logging


logger = logging.getLogger("monitoring_schedule.lambda_handler")


def describe_monitoring_schedules(endpoint_name):
    sagemaker_client = boto3.client("sagemaker")
    schedules = []
    response = sagemaker_client.list_monitoring_schedules(EndpointName=endpoint_name)[
        "MonitoringScheduleSummaries"
    ]
    for item in response:
        name = item["MonitoringScheduleName"]
        schedule = {
            "Name": name,
            "Type": item["MonitoringType"],
        }
        schedules.append(schedule)
    return schedules


def describe_monitoring_schedule(endpoint_name, monitoring_type):
    found = False
    schedules = describe_monitoring_schedules(endpoint_name)
    for schedule in schedules:
        if schedule["Type"] == monitoring_type:
            found = True
    return found


def describe_data_monitoring_schedule(endpoint_name):
    return describe_monitoring_schedule(endpoint_name, "DataQuality")


def describe_model_monitoring_schedule(endpoint_name):
    return describe_monitoring_schedule(endpoint_name, "ModelQuality")


def lambda_handler(event, context):
    logger.info("Event of endpoint state change is received.")
    try:
        boto3.client("sagemaker").describe_endpoint(EndpointName=os.environ["ENDPOINT"])
    except botocore.exceptions.ClientError as e:
        ep = os.environ["ENDPOINT"]
        logger.error(f"Endpoint {ep} wasn't found.")
        raise e
    else:
        # DATA QUALITY MONITORING SCHEDULE
        data_mon_found = describe_data_monitoring_schedule(os.environ["ENDPOINT"])
        if not data_mon_found:
            logger.info("Data quality monitor schedule will be created.")

            def create_data_monitoring_schedule():
                data_monitor = DefaultModelMonitor(
                    instance_type=os.environ["INSTANCE_TYPE"],
                    instance_count=1,
                    max_runtime_in_seconds=3600,
                    role=os.environ["SM_EXEC_ROLE"],
                )
                data_monitor.create_monitoring_schedule(
                    monitor_schedule_name="penguins-data-monitoring-schedule",
                    endpoint_input=os.environ["ENDPOINT"],
                    record_preprocessor_script=os.path.join(
                        os.environ["MONITORING_PREPROCESSING_SCRIPT"],
                        "data_quality_monitoring_preprocessing.py",
                    ),
                    statistics=os.path.join(
                        os.environ["DATA_QUALITY_LOCATION"], "statistics.json"
                    ),
                    constraints=os.path.join(
                        os.environ["DATA_QUALITY_LOCATION"], "constraints.json"
                    ),
                    schedule_cron_expression=CronExpressionGenerator.hourly(),
                    output_s3_uri=os.environ["DATA_QUALITY_LOCATION"],
                    enable_cloudwatch_metrics=True,
                )
                time.sleep(10)
                data_monitor.start_monitoring_schedule()

            th_data_mon = threading.Thread(target=create_data_monitoring_schedule)
            th_data_mon.start()

        else:
            logger.info("There is already data quality monitor schedule")

        # MODEL QUALITY MONITOR SCHEDULE
        model_mon_found = describe_model_monitoring_schedule(os.environ["ENDPOINT"])
        if not model_mon_found:
            logger.info("Model quality monitor schedule will be created.")

            def create_model_monitoring_schedule():
                model_monitor = ModelQualityMonitor(
                    instance_type=os.environ["INSTANCE_TYPE"],
                    instance_count=1,
                    max_runtime_in_seconds=1800,
                    volume_size_in_gb=20,
                    role=os.environ["SM_EXEC_ROLE"],
                )
                model_monitor.create_monitoring_schedule(
                    monitor_schedule_name="penguins-model-monitoring-schedule",
                    endpoint_input=EndpointInput(
                        endpoint_name=os.environ["ENDPOINT"],
                        # The first attribute is the prediction made
                        # by the model. For example, here is a
                        # potential output from the model:
                        # [Adelie,0.977324724\n]
                        inference_attribute="0",
                        destination="/opt/ml/processing/input_data",
                    ),
                    problem_type="MulticlassClassification",
                    ground_truth_input=os.environ["GROUND_TRUTH_LOCATION"],
                    constraints=os.path.join(
                        os.environ["MODEL_QUALITY_LOCATION"], "constraints.json"
                    ),
                    schedule_cron_expression=CronExpressionGenerator.hourly(),
                    output_s3_uri=os.environ["MODEL_QUALITY_LOCATION"],
                    enable_cloudwatch_metrics=True,
                )
                # Let's give SageMaker some time to process the
                # monitoring job before we start it.
                time.sleep(10)
                model_monitor.start_monitoring_schedule()

            th_model_mon = threading.Thread(target=create_model_monitoring_schedule)
            th_model_mon.start()

        else:
            logger.info("There is already model quality monitor schedule")

    # in case the threads are created, wait until they are done.
    if not data_mon_found:
        th_data_mon.join()
    if not model_mon_found:
        th_model_mon.join()
