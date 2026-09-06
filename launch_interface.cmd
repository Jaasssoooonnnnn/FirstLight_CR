@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0native_runner\launch_interface.ps1" %*
exit /b %ERRORLEVEL%
