#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt -q

echo ""
echo "Starting ML Smell Activity Analyzer..."
echo "Open: http://localhost:8000"
echo ""

python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload \
  --reload-dir backend --reload-dir frontend
