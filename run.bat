@echo off
echo Starting Energy Consumption Analysis System...
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Starting Streamlit application...
echo Navigate to http://localhost:8501 in your browser
echo.
streamlit run app.py
pause
