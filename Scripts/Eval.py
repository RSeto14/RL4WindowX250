# buffer torch実装
import argparse

import os
import datetime
import pytz
import sys
import numpy as np
import itertools
import torch
import csv

import random
import glob

script_dir = os.path.dirname(__file__)    
parent_dir1 = os.path.dirname(script_dir)
parent_dir2 = os.path.dirname(parent_dir1)
sys.path.append(parent_dir2)

# script_name = os.path.basename(__file__)[: -len(".py")]


from SoftActorCritic.SAC import SAC_Eval
from Env import ReachingTask
from cfg import Config
from Save_Load_cfg import dataclass_to_json, json_to_dataclass

script_name = os.path.basename(__file__)[: -len(".py")]


def Parse_args():
    parser = argparse.ArgumentParser(description='SAC eval')
   
    parser.add_argument("--train_log", type=str, default=r"C:\Users\hayas\RL4WindowX250\Log\241010_154950",help="train log dir name")
    
    
    parser.add_argument("--gpu", type=int, default=-1, help="run on CUDA -1:CPU")
    parser.add_argument("--seed", type=int, default=12345, help="seed")
    
    parser.add_argument("--headless", type=bool, default=True   , help="headless")
    parser.add_argument("--cap", type=bool, default=True,help="capture video")
    
    parser.add_argument("--net", type=int, default=0,help="Networks(episode) or 0 (best.pt)")
    
    parser.add_argument("--n_ep", type=int, default=1, help="num episodes")

    
    parser.add_argument("--alog", type=bool, default=True,help="action log")
    parser.add_argument("--olog", type=bool, default=True,help="observation log")
    
    args = parser.parse_args()
    
    return args



def main(args):
    
    # Networks
    if args.net is None:
        network_files = glob.glob(f"{args.train_log}/Networks/episode_*.pt")
        if network_files:
            latest_network = max(network_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            networks = latest_network
        else:
            raise FileNotFoundError("No network files found.")
    elif args.net == 0:
            networks = f"{args.train_log}/Networks/best.pt"
    else:
        networks = f"{args.train_log}/Networks/episode_{args.net}.pt"
    
    # make log dir
    timezone = pytz.timezone('Asia/Tokyo')
    start_datetime = datetime.datetime.now(timezone)    
    start_formatted = start_datetime.strftime("%y%m%d_%H%M%S")
    test_log_dir = f"{args.train_log}/Test/{start_formatted}"
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)
        os.makedirs(f"{test_log_dir}/CSVs")


    # train log
    with open(f"{test_log_dir}/log.txt", 'w') as file:
        file.write(f'Networks: {networks}\n')
        start = start_datetime.strftime("%y/%m/%d %H:%M:%S")
        #PID
        pid = os.getpid()
        file.write(f'Process ID: {pid}\n')
        file.write(f'Start: {start}\n')
    
    with open(f"{test_log_dir}/CSVs/rewards.csv", 'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([["Episode","Episode_Steps","Rewards","Ave_Rewards","Time"]])
        writer.writerows([[0,]])
    
    # with open(f"{test_log_dir}/CSVs/action.csv", 'w',newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows([["mu_FR","mu_FL","mu_RR","mu_RL",
    #                        "omega_FR","omega_FL","omega_RR","omega_RL",
    #                        "psi_FR","psi_FL","psi_RR","psi_RL",]])
        
    # Set config
    cfg_data = json_to_dataclass(file_path=f"{args.train_log}/config.json")
    cfg = Config(**cfg_data)
    cfg.seed = args.seed
    cfg.gpu = args.gpu
    
    dataclass_to_json(cfg,file_path=f"{test_log_dir}/config.json")
    
    
    
    # Seed
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device
    if cfg.gpu >= 0:
        visible_device = cfg.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))
        cfg.gpu = 0
    
    # Environment
    env = ReachingTask(headless=args.headless,cfg=cfg,capture=args.cap)

    # Agent
    agent = SAC_Eval(env.observation_dim, env.action_dim, cfg)

    agent.load_checkpoint(ckpt_path=networks,evaluate=True)


    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset(video_folder=f"{test_log_dir}/Videos")


        
        print(f"Episode:{i_episode}")
        
        if args.alog:
            with open(f"{test_log_dir}/CSVs/action_{i_episode}.csv", 'w',newline='') as file:
                writer = csv.writer(file)
                writer.writerows([["step","q1","q2","q3","q4","q5","q6"]])
                
        if args.olog:
            with open(f"{test_log_dir}/CSVs/obs_{i_episode}.csv", 'w',newline='') as file:
                writer = csv.writer(file)
                writer.writerows([["step","q1","q2","q3","q4","q5","q6","dq1","dq2","dq3","dq4","dq5","dq6","target x","target y","target z","end effector x","end effector y","end effector z"]])

        while not done:
            
            # print(f"gain_factor:{env.gain_factor}")
            
            action = agent.select_action(state,evaluate=True)  # Sample action from policy
            
            if args.alog:
                with open(f"{test_log_dir}/CSVs/action_{i_episode}.csv", 'a',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows([[episode_steps]+action.tolist()])
            
            if args.olog:
                with open(f"{test_log_dir}/CSVs/obs_{i_episode}.csv", 'a',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode_steps] + state.tolist())
        
            next_state, reward, termination, truncation, info = env.step(action) # Step
            episode_steps += 1
            episode_reward += reward

            state = next_state
            
            done = termination or truncation
            

        with open(f"{test_log_dir}/CSVs/rewards.csv", 'a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[i_episode, episode_steps, round(episode_reward, 4), round(episode_reward/episode_steps, 4),datetime.datetime.now(timezone)-start_datetime]])
        
        if i_episode >= args.n_ep:
            env.reset(video_folder=f"{test_log_dir}/Videos")
            break
    
    with open(f"{test_log_dir}/log.txt", 'a') as file:
        finish_datetime = datetime.datetime.now(timezone)
        finish = finish_datetime.strftime("%y/%m/%d %H:%M:%S")
        file.write(f'Finished: {finish}\n')
        file.write(f'It takes {finish_datetime - start_datetime}\n')

if __name__ == '__main__':
    args = Parse_args()
    main(args)