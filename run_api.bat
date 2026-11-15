@echo off
title EcoSort AI - FastAPI Server
echo ========================================================
echo   KHOI DONG ECOSORT AI FASTAPI SERVER
echo   Tai lieu Swagger UI: http://127.0.0.1:8000/docs
echo ========================================================
.\venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
