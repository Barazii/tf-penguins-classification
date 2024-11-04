import random
from datetime import datetime, timezone
import pandas as pd
from sagemaker.s3 import S3Uploader
import json


records = []
GROUND_TRUTH_LOCATION = "s3://penguinsmlschool/monitoring/groundtruth"
data = pd.read_csv("../data/penguins_cleaned.csv")
for inference_id in range(len(data)):
    random.seed(inference_id)

    records.append(
        json.dumps(
            {
                "groundTruthData": {
                    # For testing purposes, we will generate a random
                    # label for each request.
                    "data": random.choice(["Adelie", "Chinstrap", "Gentoo"]),
                    "encoding": "CSV",
                },
                "eventMetadata": {
                    # This value should match the id of the request
                    # captured by the endpoint.
                    "eventId": str(inference_id),
                },
                "eventVersion": "0",
            },
        ),
    )

groundtruth_payload = "\n".join(records)
upload_time = datetime.now(tz=timezone.utc)
uri = f"{GROUND_TRUTH_LOCATION}/{upload_time:%Y/%m/%d/%H/%M%S}.jsonl"
S3Uploader.upload_string_as_file_body(groundtruth_payload, uri)