import boto3
import json
from time import sleep
from sagemaker.model_monitor import MonitoringExecution
from sagemaker.session import Session
from sagemaker.s3 import S3Downloader
import os
import time


sagemaker_client = boto3.client("sagemaker")
sagemaker_session = Session()


def describe_monitoring_schedules(endpoint_name):
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

        description = sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=name
        )

        schedule["Status"] = description["MonitoringScheduleStatus"]

        last_execution_status = description["LastMonitoringExecutionSummary"][
            "MonitoringExecutionStatus"
        ]

        schedule["Last Execution Status"] = last_execution_status
        schedule["Last Execution Date"] = str(description["LastMonitoringExecutionSummary"]["LastModifiedTime"])

        if last_execution_status == "CompletedWithViolations":
            processing_job_arn = description["LastMonitoringExecutionSummary"][
                "ProcessingJobArn"
            ]
            execution = MonitoringExecution.from_processing_arn(
                sagemaker_session=sagemaker_session,
                processing_job_arn=processing_job_arn,
            )
            execution_destination = execution.output.destination

            violations_filepath = os.path.join(
                execution_destination, "constraint_violations.json"
            )
            violations = json.loads(S3Downloader.read_file(violations_filepath))[
                "violations"
            ]

            schedule["Violations"] = violations

        schedules.append(schedule)

    return schedules

def describe_monitoring_schedule(endpoint_name, monitoring_type):
    found = False

    schedules = describe_monitoring_schedules(endpoint_name)
    for schedule in schedules:
        if schedule["Type"] == monitoring_type:
            found = True
            print(json.dumps(schedule, indent=2))

    if not found:
        print(f"There's no {monitoring_type} Monitoring Schedule.")
    
    return found

def describe_data_monitoring_schedule(endpoint_name):
    return describe_monitoring_schedule(endpoint_name, "DataQuality")

def describe_model_monitoring_schedule(endpoint_name):
    return describe_monitoring_schedule(endpoint_name, "ModelQuality")

def delete_monitoring_schedule(endpoint_name, monitoring_type):
    attempts = 30
    found = False

    response = sagemaker_client.list_monitoring_schedules(EndpointName=endpoint_name)[
        "MonitoringScheduleSummaries"
    ]
    for item in response:
        if item["MonitoringType"] == monitoring_type:
            found = True
            
            summary = sagemaker_client.describe_monitoring_schedule(
                MonitoringScheduleName=item["MonitoringScheduleName"]
            )
            status = summary["MonitoringScheduleStatus"]

            if status == "Scheduled" and "LastMonitoringExecutionSummary" in summary and "MonitoringExecutionStatus" in summary["LastMonitoringExecutionSummary"]:
                status = summary["LastMonitoringExecutionSummary"]["MonitoringExecutionStatus"]

            while status in ("Pending", "InProgress") and attempts > 0:
                attempts -= 1
                print(
                    f"Monitoring schedule status: {status}. Waiting for it to finish."
                )
                sleep(30)

                status = sagemaker_client.describe_monitoring_schedule(
                    MonitoringScheduleName=item["MonitoringScheduleName"]
                )["MonitoringScheduleStatus"]

            if status not in ("Pending", "InProgress"):
                sagemaker_client.delete_monitoring_schedule(
                    MonitoringScheduleName=item["MonitoringScheduleName"]
                )
                print("Monitoring schedule deleted.")
            else:
                print("Waiting for monitoring schedule timed out")

    if not found:
        print(f"There's no {monitoring_type} Monitoring Schedule.")

def delete_data_monitoring_schedule(endpoint_name):
    delete_monitoring_schedule(endpoint_name, "DataQuality")
    time.sleep(10)

def delete_model_monitoring_schedule(endpoint_name):
    delete_monitoring_schedule(endpoint_name, "ModelQuality")
    time.sleep(10)