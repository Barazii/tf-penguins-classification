# Penguins Classification with AWS SageMaker

## Overview

This project demonstrates a complete machine learning pipeline for classifying penguin species using AWS SageMaker. By leveraging SageMaker's managed services, the workflow covers data preprocessing, model training, deployment, and continuous monitoring—all orchestrated for scalability and automation on the AWS cloud.

The dataset is based on the classic Palmer Penguins dataset, providing a practical alternative to the Iris dataset for classification tasks. The model used in this project is just a placeholder that can be replaced with a better more powerful model. The goal is to build the machine learning pipeline around the model, not the model itself.

## Conceptual design

AWS SageMaker is used for orchestrating the machine learning pipeline, allowing for easy automation, scalability, and integration with other AWS services.

![Pipeline Diagram](images/penguins-classification-pipeline-2.png)

**Key Components:**
- **Data Processing:** Cleans and prepares the penguin data for training.
- **Model Training:** Trains a machine learning model using SageMaker’s managed infrastructure.
- **Deployment:** Utilizes AWS Lambda functions for automated model deployment and to initiate model monitoring schedules.
- **Monitoring:** Continuously monitors deployed models for data drift and performance.

## Run the pipeline
The pipeline could run locally, but isn't recommended as Sagemaker is primarily made to run machine learning systems on AWS cloud.

1. **Set up your environment:**
   - Create and activate a virtual environment.
   - Install dependencies from `requirements.txt`.

2. **Run the pipeline:**
   ```bash
   bash run_pipeline.sh
   
Everything in the pipeline will run from the main program file, including the two AWS Lambda functions used for the model deployment and starting the monitoring schedules. \
The tests are automated using a GitHub action. The tests can still be run locally though:
```
bash run_tests.sh
