# Deep Reinforcement Learning for WindowX 250

## 環境構築

### 仮想環境

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 必要ライブラリのインストール

```bash
python.exe -m pip install --upgrade pip
pip install -r src/requirements.txt
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### Git の使い方メモ

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/RSeto14/RL4WindowX250.git
git push -u origin main
```

