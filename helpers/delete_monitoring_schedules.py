from monitoring_helpers import delete_data_monitoring_schedule, delete_model_monitoring_schedule
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv()
    delete_data_monitoring_schedule(os.environ["ENDPOINT"])
    delete_model_monitoring_schedule(os.environ["ENDPOINT"])