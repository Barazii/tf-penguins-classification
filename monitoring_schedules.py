from constants import *
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DefaultModelMonitor,
    EndpointInput,
)
from helpers.monitoring_helpers import *
from sagemaker.model_monitor import ModelQualityMonitor
import botocore
import time
import argparse
import threading


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True, default=ENDPOINT)
    args = parser.parse_args()
    endpoint = args.endpoint

    try:
        boto3.client("sagemaker").describe_endpoint(EndpointName=endpoint)
    except botocore.exceptions.ClientError:
        print(f"Endpoint {ENDPOINT} wasn't found.")
    else:
        # DATA QUALITY MONITORING SCHEDULE
        found = describe_data_monitoring_schedule(ENDPOINT)
        if not found:
            print("Data quality monitor schedule will be created.")
            def create_data_monitoring_schedule():
                data_monitor = DefaultModelMonitor(
                    instance_type=instance_type,
                    instance_count=1,
                    max_runtime_in_seconds=3600,
                    role=role,
                )
                data_monitor.create_monitoring_schedule(
                    monitor_schedule_name="penguins-data-monitoring-schedule",
                    endpoint_input=ENDPOINT,
                    record_preprocessor_script="code/data_quality_monitoring_preprocessing.py",
                    statistics=f"{DATA_QUALITY_LOCATION}/statistics.json",
                    constraints=f"{DATA_QUALITY_LOCATION}/constraints.json",
                    schedule_cron_expression=CronExpressionGenerator.now(),
                    output_s3_uri=DATA_QUALITY_LOCATION,
                    enable_cloudwatch_metrics=True,
                    data_analysis_start_time="-PT1H",
                    data_analysis_end_time="-PT0H",
                )
                time.sleep(10)
                data_monitor.start_monitoring_schedule()
            threading.Thread(target=create_data_monitoring_schedule).start()
        else:
            print("There is already data quality monitor schedule")

        # MODEL QUALITY MONITOR SCHEDULE
        found = describe_model_monitoring_schedule(ENDPOINT)
        if not found:
            print("Model quality monitor schedule will be created.")
            def create_model_monitoring_schedule():
                model_monitor = ModelQualityMonitor(
                    instance_type=instance_type,
                    instance_count=1,
                    max_runtime_in_seconds=1800,
                    volume_size_in_gb=20,
                    role=role,
                )
                model_monitor.create_monitoring_schedule(
                    monitor_schedule_name="penguins-model-monitoring-schedule",
                    endpoint_input=EndpointInput(
                        endpoint_name=ENDPOINT,
                        # The first attribute is the prediction made
                        # by the model. For example, here is a
                        # potential output from the model:
                        # [Adelie,0.977324724\n]
                        inference_attribute="0",
                        destination="/opt/ml/processing/input_data",
                    ),
                    problem_type="MulticlassClassification",
                    ground_truth_input=GROUND_TRUTH_LOCATION,
                    constraints=f"{MODEL_QUALITY_LOCATION}/constraints.json",
                    schedule_cron_expression=CronExpressionGenerator.hourly(),
                    output_s3_uri=MODEL_QUALITY_LOCATION,
                    enable_cloudwatch_metrics=True,
                )
                # Let's give SageMaker some time to process the
                # monitoring job before we start it.
                time.sleep(10)
                model_monitor.start_monitoring_schedule()
            thread = threading.Thread(target=create_model_monitoring_schedule)
            thread.start()
            thread.join()
        else:
            print("There is already model quality monitor schedule")
