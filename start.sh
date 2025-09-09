#!/bin/bash
# start.sh
pip install -r requirements.txt
playwright install chromium
uvicorn main:app --host 0.0.0.0 --port $PORT
