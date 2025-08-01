@echo off
echo Starting Deep Planning LangGraph Web Application...
echo.

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found!
    echo Please copy .env.example to .env and configure your API keys.
    pause
    exit /b 1
)

REM --- Activate Conda and run the app ---

REM Path to conda activate script. This is a common location.
set CONDA_ACTIVATE_SCRIPT=C:\Code\anaconda3\Scripts\activate.bat

if not exist "%CONDA_ACTIVATE_SCRIPT%" (
    echo Conda activation script not found at %CONDA_ACTIVATE_SCRIPT%
    echo Please update the path in this script if your Anaconda installation is different.
    pause
    exit /b 1
)

REM Activate the base conda environment and run the web app
call %CONDA_ACTIVATE_SCRIPT% base

if %errorlevel% neq 0 (
    echo Failed to activate conda environment.
    pause
    exit /b 1
)

echo Conda environment activated. Starting application...
echo.

REM Change to the parent directory so that the module can be found
cd ..

python -m ProductDesigner.web_app

if %errorlevel% neq 0 (
    echo Failed to start the application.
    pause
    exit /b 1
)

echo.
echo Application is running. Press Ctrl+C to stop the server.
