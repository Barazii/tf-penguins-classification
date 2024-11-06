source ~/.sagemaker/bin/activate
source .env
endpoint=${1:-"$ENDPOINT"}
python3 monitoring_schedules.py --endpoint "$endpoint"