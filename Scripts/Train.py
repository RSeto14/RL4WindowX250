# buffer torch実装
import argparse

import os
import pickle
import datetime
import pytz
import sys
import numpy as np
import itertools
import torch
import csv

import random

script_dir = os.path.dirname(__file__)    
parent_dir1 = os.path.dirname(script_dir)
parent_dir2 = os.path.dirname(parent_dir1)
sys.path.append(parent_dir2)

# script_name = os.path.basename(__file__)[: -len(".py")]


from SoftActorCritic.SAC import SAC
from SoftActorCritic.ReplayMemory import ReplayMemory
from Env import ReachingTask
from cfg import Config
from Save_Load_cfg import dataclass_to_json




def Parse_args():
    parser = argparse.ArgumentParser(description='SAC train')
    
    parser.add_argument("--gpu", type=int, default=-1, help="run on CUDA (default: 0) cpu: -1")
    parser.add_argument("--seed", type=int, default=1234567, help="seed")


    args = parser.parse_args()
    
    return args



def main(args):
    
    cfg = Config(seed=args.seed, gpu=args.gpu)
    
    
    
    # make log dir
    timezone = pytz.timezone('Asia/Tokyo')
    start_datetime = datetime.datetime.now(timezone)    
    start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
    train_log_dir = f"{parent_dir1}/Log/{start_formatted}"
    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
        os.makedirs(f"{train_log_dir}/CSVs")
        os.makedirs(f"{train_log_dir}/Networks")
        
    

    # train log
    with open(f"{train_log_dir}/log.txt", 'w') as file:
        start = start_datetime.strftime("%y/%m/%d %H:%M:%S")
        #PID
        pid = os.getpid()
        file.write(f'Process ID: {pid}\n')
        file.write(f'Start: {start}\n')
    
    
        
    
    
    with open(f"{train_log_dir}/CSVs/rewards.csv", 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([["Episode","Episode_Steps","Total Steps","Rewards","Time"]])
        writer.writerows([[0,]])
        
    with open(f"{train_log_dir}/CSVs/log.csv", 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([["Global Steps","Critic 1 loss", "Critic 2 loss", "Policy loss", "Entropy loss", "alpha"]])
        writer.writerows([[0,]])
    
    dataclass_to_json(cfg,file_path=f"{train_log_dir}/config.json")
    
    
    # Seed
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device
    if cfg.gpu >=0:
        visible_device = cfg.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
        cfg.gpu = 0
    
    # Environment
    env = ReachingTask(cfg=cfg)

    # Agent
    agent = SAC(env.observation_dim, env.action_dim, cfg)

    # Memory
    memory = ReplayMemory(cfg.replay_size, env.observation_dim, env.action_dim, seed, gpu=cfg.gpu)
        
    # Training Loop
    total_numsteps = 0
    updates = 0
    max_episode_rewards = -10000
    best_episode = 0
    
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        
        
        state = env.reset()
        

        while not done:
            if cfg.start_steps > total_numsteps:
                
                
                action = np.random.normal(0, 1, env.action_dim)  # 平均0、標準偏差1の正規分布に基づく値を生成
                action = np.tanh(action)
                
            else:
                action = agent.select_action(state)  # Sample action from policy

                if len(memory) > cfg.batch_size:
                    # Number of updates per step in environment
                    if total_numsteps%cfg.updates_interval==0:
                        for _ in range(cfg.updates_per_step):
                            # Update parameters of all the networks
                            
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, cfg.batch_size, updates)
                            updates += 1
                    if total_numsteps%cfg.log_interval==0:
                        with open(f"{train_log_dir}/CSVs/log.csv", 'a',newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows([[total_numsteps, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha]])
                        
            
            
            next_state, reward, termination, truncation, info = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            
            mask = int(not termination)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state
            
            done = termination or truncation
            
            
        if total_numsteps >= cfg.start_steps and episode_reward > max_episode_rewards:
            agent.save_checkpoint(ckpt_dir=f"{train_log_dir}/Networks", file_name=f"best.pt" )
            max_episode_rewards = episode_reward
            best_episode = i_episode
        if total_numsteps >= cfg.start_steps and i_episode%10 == 0:
            agent.save_checkpoint(ckpt_dir=f"{train_log_dir}/Networks", file_name=f"episode_{i_episode}.pt" )
            

        with open(train_log_dir + "/CSVs/rewards.csv", 'a',newline='') as file:
            writer = csv.writer(file)
            # writer.writerows([[i_episode, episode_steps, total_numsteps, round(episode_reward, 2)]])
            writer.writerows([[i_episode, episode_steps, total_numsteps, round(episode_reward, 4), datetime.datetime.now(timezone)-start_datetime]])
                
        if total_numsteps >= cfg.num_steps:
            break
    
    memory.save_buffer(save_dir=train_log_dir + "/Buffer")
    
    with open(f"{train_log_dir}/log.txt", 'a') as file:
        finish_datetime = datetime.datetime.now(timezone)
        finish = finish_datetime.strftime("%y/%m/%d %H:%M:%S")
        file.write(f'Finished: {finish}\n')
        file.write(f'It takes {finish_datetime - start_datetime}\n')      
        file.write(f'Best Episode: {best_episode}\n')  

if __name__ == '__main__':
    args = Parse_args()
    main(args)