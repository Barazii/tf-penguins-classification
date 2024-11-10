from helpers.monitoring_helpers import delete_data_monitoring_schedule, delete_model_monitoring_schedule
from constants import ENDPOINT


if __name__ == "__main__":
    delete_data_monitoring_schedule(ENDPOINT)
    delete_model_monitoring_schedule(ENDPOINT)