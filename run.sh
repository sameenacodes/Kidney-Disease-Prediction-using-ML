#!/usr/bin/env bash
# Linux / macOS quick start
set -e
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -r requirements.txt
[ -f models/ckd_model.joblib ] || python ml/train.py
python app.py
