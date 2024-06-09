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
ppc_base_directory = "/opt/ml/preprocessing"    # preprocessing container base directory 
input_data_directory = f"{ppc_base_directory}/data"
data_splits_directory = f"{ppc_base_directory}/data-splits"
model_directory = f"{ppc_base_directory}/model"
transformers_directory = f"{ppc_base_directory}/transformers"
baseline_directory = f"{ppc_base_directory}/baseline"