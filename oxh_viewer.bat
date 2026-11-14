@echo off
rem OXH Viewer launcher for Windows (double-click association for .oxh)
if exist "%~dp0venv\Scripts\activate.bat" call "%~dp0venv\Scripts\activate.bat"
python "%~dp0oxh_viewer.py" %*
