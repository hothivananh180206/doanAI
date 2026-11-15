@echo off
cd /d "%~dp0"
title He Thong AI Phan Loai Rac - Streamlit (100%% Python & YOLO)
echo ============================================================
echo   DANG KHOI CHAY UNG DUNG WEB PHAN LOAI RAC BANG AI (YOLO)
echo ============================================================

set STREAMLIT_GATHER_USAGE_STATS=false

if exist "d:\doanAI\venv\Scripts\streamlit.exe" (
    "d:\doanAI\venv\Scripts\streamlit.exe" run "d:\doanAI\app.py" --browser.gatherUsageStats false
) else (
    python -m streamlit run "d:\doanAI\app.py" --browser.gatherUsageStats false
)
pause
