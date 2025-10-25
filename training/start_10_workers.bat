@echo off
echo ============================================
echo  TSUNAMI SURGE - 10 Worker Training
echo  Starting 10 parallel training instances
echo ============================================

set REDIS_IP=localhost
set WORKER_COUNT=10

for /L %%i in (1,1,%WORKER_COUNT%) do (
    echo Starting Worker %%i...
    start "Tsunami Worker %%i" cmd /k "..\training_env\Scripts\python.exe worker.py tsunami-worker-%%i %REDIS_IP% None"
    timeout /t 2 /nobreak > nul
)

echo.
echo ✅ All %WORKER_COUNT% workers started!
echo 🌊 Tsunami Surge training in progress...
echo.
pause
