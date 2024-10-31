from sagemaker.serializers import JSONSerializer, CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
import boto3
import json
import pandas as pd

ENDPOINT = "penguins-endpoint"

def generate_fake_traffic():
    """Generate fake traffic to the endpoint."""
    sagemaker_session = boto3.Session().client('sagemaker')
    predictor = Predictor(endpoint_name=ENDPOINT, serializer=CSVSerializer(),
                        #   deserializer=JSO
            #   sagemaker_session=sagemaker_session
                )
    data = pd.read_csv("data/penguins_cleaned.csv")
    data = data.drop(columns="species", axis=1)
    
    try:
        for index, row in data.iterrows():
            payload = ",".join([str(x) for x in row.to_list()])
            print(payload)
            response = predictor.predict(
                payload,
                # initial_args={"ContentType": "text/csv", "Accept": "text/csv"},
                # The `inference_id` field is important to match
                # it later with a corresponding ground-truth label.
                inference_id=str(index),
            )
            response = json.loads(response.decode("utf-8"))
            print(response)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    generate_fake_traffic()