import time
import boto3
import constants


if __name__ == "__main__":
    sagemaker_client = boto3.client("sagemaker")
    response = sagemaker_client.list_monitoring_schedules(EndpointName=constants.ENDPOINT)
    if not response["MonitoringScheduleSummaries"]:
        for i in response["MonitoringScheduleSummaries"]:
            name = i["MonitoringScheduleName"]
            sagemaker_client.delete_monitoring_schedule(MonitoringScheduleName=name)
        time.sleep(10)