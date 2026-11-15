@echo off
REM Windows quick start
cd /d "%~dp0"
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate
pip install -r requirements.txt
if not exist models\ckd_model.joblib python ml\train.py
python app.py
