# Dockerfile
FROM public.ecr.aws/lambda/python:3.10
# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# Copy handler code
COPY lambda_handler.py .
# Define Lambda function handler
CMD ["lambda_handler.lambda_handler"]