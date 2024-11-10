# Dockerfile
FROM public.ecr.aws/lambda/python:3.10
# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Copy handler code
COPY monitor_schedule_lambda_handler.py .
# Define Lambda function handler
CMD ["monitor_schedule_lambda_handler.lambda_handler"]