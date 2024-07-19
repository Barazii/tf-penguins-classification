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
MODEL_GROUP_NAME = "penguins-classifier"
FEATURE_COLUMNS = [
    "island",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]
TARGET_CATEGORIES = ['Biscoe', 'Dream', 'Torgersen']