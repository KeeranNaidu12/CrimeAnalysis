#!/bin/bash
cd /home/site/wwwroot
pip install -r requirements.txt
cd crime_prediction_api
gunicorn app:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 600
