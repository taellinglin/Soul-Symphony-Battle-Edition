@echo off
setlocal
cd /d "%~dp0"
python server_app.py 8383
endlocal
