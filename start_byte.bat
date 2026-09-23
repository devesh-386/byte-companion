@echo off
rem Starts Byte (desktop companion) without a console window.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m companion.ui
