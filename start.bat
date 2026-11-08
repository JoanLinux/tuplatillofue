@echo off
echo Iniciando TuPlatilloFue...
cd /d %~dp0
..\\.venv\Scripts\pip install deepface tf-keras --quiet
..\\.venv\Scripts\python.exe server.py
pause
