@echo off
REM Double-click to open the Jobe chat window (always on top, drives a browser).
cd /d "%~dp0.."
REM Machine-specific settings (for example JUNO_STT_SCRIPT) go in desktop/local.cmd, which git ignores.
if exist "%~dp0local.cmd" call "%~dp0local.cmd"
start "" /b ".venv\Scripts\pythonw.exe" desktop\jobe_chat.py
