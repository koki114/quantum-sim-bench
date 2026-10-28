@echo off
chcp 65001 >nul
rem Windows 用の起動スクリプト。PowerShell の実行ポリシーの影響を受けない(venv の Python を直接使う)。
rem 使い方: このファイルをダブルクリック、または cmd / PowerShell で  start_app.bat
cd /d "%~dp0"
set PY=
if exist ".venv-clean\Scripts\python.exe" set PY=.venv-clean\Scripts\python.exe
if not defined PY if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY (
  echo 仮想環境が見つかりません。README の「環境構築」を先に実行してください。
  pause
  exit /b 1
)
"%PY%" -c "import streamlit" 2>nul
if errorlevel 1 (
  echo streamlit が入っていません。次を実行してください: %PY% -m pip install -r requirements.txt
  pause
  exit /b 1
)
echo ブラウザで http://localhost:8501 を開いてください(自動で開く場合もあります)。終了は Ctrl+C。
"%PY%" -m streamlit run app\streamlit_app.py
pause
