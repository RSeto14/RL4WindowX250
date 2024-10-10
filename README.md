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
```

#### CUDAを使う場合

CUDAのバージョンに合わせてインストール

[PyTorch](https://pytorch.org/)

Ex)

```bash
pip install torch==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## 設定ファイル

[cfg.py](./Scripts/cfg.py)

SACの設定

```python
    gpu: int = -1
    seed: int = 123456
    
    # SAC
    policy: str = "Gaussian"
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 0.0001
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    batch_size: int = 256
    num_steps: int = 100000
    hidden_size: List[int] = field(default_factory=lambda: [512, 256, 128]) # 512, 256, 128
    updates_interval: int = 1
    updates_per_step: int = 1
    log_interval: int = 1000 # episode
    # start_steps: int = 100000
    start_steps: int = 50000
    target_update_interval: int = 1
    replay_size: int = 100000
```

環境の設定

```python
    # Env
    step_dt_per_mujoco_dt: int = 10
    mujoco_dt: float = 0.002

```

タスクの設定

```python
    # ReachingTask
    max_steps: int = 3000
    target_change_interval: int = 2
    target_pos_min: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.5])
    target_pos_max: List[float] = field(default_factory=lambda: [0.0, -0.3, 0.2])
    action_space_min: List[float] = field(default_factory=lambda: [-1, -1, -1, -1, -1, -1])
    action_space_max: List[float] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, ])

    # Reward
    reward_pos: float = 1.0
    reward_frc: float = -0.01
```
## 環境

[Env.py](./Scripts/Env.py)

#### WindowX250Env()

アームの基本のクラス

#### ReachingTask()

アームの基本のクラスを継承したリーチングタスク用のクラス

## 学習

[Train.py](./Scripts/Train.py)

gpu or cpuとseed値の設定
```python 

def Parse_args():
    parser = argparse.ArgumentParser(description='SAC train')
    
    parser.add_argument("--gpu", type=int, default=-1, help="run on CUDA (default: 0) cpu: -1")
    parser.add_argument("--seed", type=int, default=1234567, help="seed")


    args = parser.parse_args()
    
    return args

```

実行例
コマンド引数を入れないとデフォルト値が採用される

```
python Train.py --gpu -1 --seed 123456 --headless True
```

## 評価

[Eval.py](./Scripts/Eval.py)

設定
```python

def Parse_args():
    parser = argparse.ArgumentParser(description='SAC eval')
   
    parser.add_argument("--train_log", type=str, default=r"C:\Users\hayas\RL4WindowX250\Log\241010_125356",help="train log dir name")
    
    
    parser.add_argument("--gpu", type=int, default=-1, help="run on CUDA -1:CPU")
    parser.add_argument("--seed", type=int, default=123456, help="seed")
    
    parser.add_argument("--headless", type=bool, default=True, help="headless")
    parser.add_argument("--cap", type=bool, default=True,help="capture video")
    
    parser.add_argument("--net", type=int, default=0,help="Networks(episode) or 0 (best.pt)")
    
    parser.add_argument("--n_ep", type=int, default=2, help="num episodes")

    
    parser.add_argument("--alog", type=bool, default=True,help="action log")
    parser.add_argument("--olog", type=bool, default=True,help="observation log")
    
    args = parser.parse_args()
    
    return args

```

実行例

```
python Eval.py --train_log C:\Users\hayas\RL4WindowX250\Log\241010_125356 --gpu -1 --seed 123456 --headless True --cap True --net 0 --n_ep 2 --alog True --olog True
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