from dotenv import load_dotenv
import os


CLEANED_DATA_PATH = "./data/penguins_cleaned.csv"
load_dotenv()
role = os.environ["ROLE"]
bucket = os.environ["BUCKET"]
s3_project_uri = f"s3://{bucket}"
instance_type = "ml.m5.xlarge"
tf_version = "2.11"
skl_version = "1.2-1"
py_version = "py39"
pc_base_directory = "/opt/ml/processing"    # processing container base directory
MODEL_PACKAGE_GROUP_NAME = "penguins-classifier"
ENDPOINT = "penguins-endpoint"
DATA_CAPTURE_DESTINATION = f"{s3_project_uri}/monitoring/data-capture"
GROUND_TRUTH_LOCATION = f"{s3_project_uri}/monitoring/groundtruth"
DATA_QUALITY_LOCATION = f"{s3_project_uri}/monitoring/data-quality"
MODEL_QUALITY_LOCATION = f"{s3_project_uri}/monitoring/model-quality"
TRANSFORM_LOCATION = f"{s3_project_uri}/monitoring/transform"
MONITORING_SCRIPT_LOCATION = "monitoring/monitoring-scripts/data_quality_preprocessing.py"
data_quality_preprocessing_uri = f"{s3_project_uri}/monitoring/monitoring-scripts/data_quality_preprocessing.py"
data_monitor_schedule_name = "data-monitoring-schedule"
model_monitor_schedule_name = "model-monitoring-schedule"