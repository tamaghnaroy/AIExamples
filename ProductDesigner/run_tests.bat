@echo off
REM Batch file to run tests for the Deep Planning LangGraph system on Windows

echo 🧪 Deep Planning LangGraph Test Runner
echo =====================================

REM Check if .env file exists
if not exist .env (
    echo ⚠️  Warning: .env file not found
    echo    Tests will use default mock values
    echo.
)

REM Check for command line arguments
if "%1"=="--install-deps" (
    echo 📦 Installing test dependencies...
    goto install_deps
)

REM Try to run with python, then py if python fails
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Using python command
    python run_tests.py %*
    goto check_result
) else (
    py --version >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Using py command
        py run_tests.py %*
        goto check_result
    ) else (
        echo ❌ Error: Python not found in PATH
        echo    Please install Python or add it to your PATH
        goto end_with_pause
    )
)

:install_deps
python --version >nul 2>&1
if %errorlevel% == 0 (
    python run_tests.py --install-deps
) else (
    py --version >nul 2>&1
    if %errorlevel% == 0 (
        py run_tests.py --install-deps
    ) else (
        echo ❌ Error: Python not found in PATH
        goto end_with_pause
    )
)
goto end_with_pause

:check_result
REM Check the exit code
if %errorlevel% == 0 (
    echo.
    echo 🎉 Test execution completed successfully!
) else (
    echo.
    echo ❌ Test execution failed with exit code %errorlevel%
    echo.
    echo 💡 If you see coverage-related errors, try:
    echo    run_tests.bat --install-deps
)
goto end_with_pause

:end_with_pause
echo.
echo Press any key to continue...
pause >nul
