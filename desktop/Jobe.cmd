@echo off
REM Double-click to open the Jobe chat window (always on top, drives a browser).
cd /d "%~dp0.."
start "" /b ".venv\Scripts\pythonw.exe" desktop\jobe_chat.py
