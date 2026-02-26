@echo off
setlocal

REM Create the temp directory that Claude Code needs
set "TMPDIR=C:\Users\lehph\AppData\Local\Temp\claude\C--Users-lehph-Documents-GitHub-BookMapOrderFlowStudies-2\tasks"
if not exist "%TMPDIR%" mkdir "%TMPDIR%"
echo Created temp dir: %TMPDIR%

REM Run Command 1: OR Reversal
echo.
echo ============================================================
echo COMMAND 1: OR Reversal
echo ============================================================
cd /d "C:\Users\lehph\Documents\GitHub\BookMapOrderFlowStudies-2"
"E:\anaconda\python.exe" scripts\run_backtest.py --strategies "Opening Range Rev" --instrument NQ > output\cmd1_result.txt 2>&1
echo Command 1 exit code: %ERRORLEVEL%

REM Run Command 2: OR Acceptance
echo.
echo ============================================================
echo COMMAND 2: OR Acceptance
echo ============================================================
"E:\anaconda\python.exe" scripts\run_backtest.py --strategies "OR Acceptance" --instrument NQ > output\cmd2_result.txt 2>&1
set CMD2_EXIT=%ERRORLEVEL%
echo Command 2 exit code: %CMD2_EXIT%

REM If command 2 failed, try fallback
if %CMD2_EXIT% NEQ 0 (
    echo Trying fallback: --strategies "Acceptance"
    "E:\anaconda\python.exe" scripts\run_backtest.py --strategies "Acceptance" --instrument NQ > output\cmd2_fallback_result.txt 2>&1
    echo Fallback exit code: %ERRORLEVEL%
)

echo.
echo All done. Results in output\cmd1_result.txt and output\cmd2_result.txt
endlocal
