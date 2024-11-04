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
DATA_CAPTURE_DESTINATION = "s3://penguinsmlschool/monitoring/data-capture"
GROUND_TRUTH_LOCATION = "s3://penguinsmlschool/monitoring/groundtruth"
DATA_QUALITY_LOCATION = "s3://penguinsmlschool/monitoring/data-quality"
MODEL_QUALITY_LOCATION = "s3://penguinsmlschool/monitoring/model-quality"
TRANSFORM_LOCATION = "s3://penguinsmlschool/monitoring/transform"
MONITORING_PREPROCESSING_SCRIPT = "s3://penguinsmlschool/monitoring/data_quality_monitoring_preprocessing.py"