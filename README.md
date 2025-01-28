## Conceptual design
AWS sagemaker is used for orchestrating the machine learning pipeline.\
![Alt Text](images/penguins-classification-pipeline-2.png)

## Run the pipeline
The pipeline could run locally, but isn't recommended as sagemaker is primarily made to run machine learning systems on AWS cloud. \
Assuming a virtual env is created and equipped with all the dependencies in requirements.txt, execute:\
```bash run_pipeline.sh 
Everything in the pipeline will run from the main program file including the 2 AWS lambda functions used for the model deployment and starting the monitoring schedules.\ 
The tests are automated using a github action. The tests can still be run locally:\
`bash run_tests.sh`
