# Create FastAPI App
1. Create Python Env with Rye
```bash
rye init .  # ryeプロジェクト初期化
rye pin 3.11  # pythonのバージョンを3.11に指定
rye sync  # 設定を反映
```

2. Add FastAPI libs
```bash
rye add fastapi uvicorn
rye sync
```

3. Create main.py
- CMD+Shift+L でAIチャットして作成
