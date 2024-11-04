from constants import *
from sagemaker.model_monitor import CronExpressionGenerator, DefaultModelMonitor
from helpers.monitoring_helpers import *


if __name__ == "__main__":
    # delete_data_monitoring_schedule(ENDPOINT)
    found = describe_data_monitoring_schedule(ENDPOINT)
    if not found:
        print("Data quality monitor schedule will be created.")
        data_monitor = DefaultModelMonitor(
            instance_type=instance_type,
            instance_count=1,
            max_runtime_in_seconds=3600,
            role=role,
        )
        data_monitor.create_monitoring_schedule(
            monitor_schedule_name="data-monitoring-schedule",
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
    print("There is already data quality monitor schedule")