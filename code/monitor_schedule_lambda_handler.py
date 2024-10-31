import boto3


sagemaker = boto3.client("sagemaker")
instance_type = "ml.m5.xlarge"


def lambda_handler(event, context):
    if "detail" in event:
        endpoint_status = event["detail"]["EndpointStatus"]

    # endpoint = os.environ["ENDPOINT"]
    endpoint = "penguins-endpoint"
    # role = os.environ["ROLE"]
    role = "arn:aws:iam::482497089777:role/service-role/AmazonSageMaker-ExecutionRole-20240203T043640"
    # data_quality_location = os.environ["DATA_QUALITY_LOCATION"]
    data_quality_location = "s3://penguinsmlschool/monitoring/data-quality"
    # monitoring_preprocessing_script = os.environ["MONITORING_PREPROCESSING_SCRIPT"]
    monitoring_preprocessing_script = "s3://penguinsmlschool/monitoring/data_quality_monitoring_preprocessing.py"

    if endpoint_status == "IN_SERVICE":
        try:
            sagemaker.create_monitoring_schedule(
                MonitoringScheduleName='data-monitoring-schedule',
                MonitoringScheduleConfig={
                    'ScheduleConfig': {
                    'ScheduleExpression': 'Hourly: cron(0 * ? * * *)'
                },
                'MonitoringJobDefinition': {
                    'BaselineConfig': {
                        'ConstraintsResource': {
                            'S3Uri': f"{data_quality_location}/constraints.json",
                        },
                        'StatisticsResource': {
                            'S3Uri': f"{data_quality_location}/statistics.json"
                        }
                    },
                    'MonitoringInputs': [
                        {
                            'EndpointInput': {
                                'EndpointName': endpoint,
                                'LocalPath': '/opt/ml/processing/input',
                            },
                        },
                    ],
                    'MonitoringOutputConfig': {
                        'MonitoringOutputs': [
                            {
                                'S3Output': {
                                    'S3Uri': data_quality_location,
                                    'LocalPath': '/opt/ml/processing/output',
                                }
                            },
                        ],
                    },
                    'MonitoringResources': {
                        'ClusterConfig': {
                            'InstanceCount': 1,
                            'InstanceType': instance_type,
                            'VolumeSizeInGB': 20,
                        }
                    },
                    'MonitoringAppSpecification': {
                        'ImageUri': '482497089777.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-model-monitor-analyzer:latest',
                        # 'RecordPreprocessorSourceUri': monitoring_preprocessing_script
                    },
                    'StoppingCondition': {
                        'MaxRuntimeInSeconds': 3600
                    },
                    'RoleArn': role,
                },
                'MonitoringType': 'DataQuality'
            },
        )
        except sagemaker.exceptions.ResourceInUse as _:
            print(f"The data monitoring schedule already exists.")
    else:
        print("Something wrong in the endpoint status.")