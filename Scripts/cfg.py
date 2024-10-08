from dataclasses import dataclass
from typing import List, Union
import json

from dataclasses import dataclass, field
from typing import List

# dataclassの定義
@dataclass
class Config:
    gpu: int = 2
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
    
    # Env
    step_dt_per_mujoco_dt: int = 10
    mujoco_dt: float = 0.002
    
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

