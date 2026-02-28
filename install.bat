@echo off
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo [2/4] Creating virtual environment...
python -m venv hf

echo [3/4] Activating environment and installing dependencies...
call hf\Scripts\activate.bat
pip install -r requirements.txt

echo [4/4] Verifying DashScope integration...
python -c "from langchain_community.chat_models import ChatTongyi; print('✅ ChatTongyi imported successfully'); llm = ChatTongyi(model='qwen-max'); print('✅ Qwen-Max model configured (API key not required for import)')"

echo.
echo 🎉 Setup complete! Run:
echo hf\Scripts\activate
echo python src/main.py --ticker AAPL --model-name qwen-max --model-provider Dashscope
pause