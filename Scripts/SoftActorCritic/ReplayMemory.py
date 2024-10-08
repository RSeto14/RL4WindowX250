import torch
import random
import os
import pickle
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, state_dim, action_dim, seed, gpu):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0
        self.device = torch.device(f"cuda:{gpu}")
        
        self.memory_size = 0

        # バッファをGPUメモリに保存
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.position] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.position] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[self.position] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.position] = torch.tensor([done], dtype=torch.float32, device=self.device)
        self.position = (self.position + 1) % self.capacity
        
        self.memory_size = min (self.memory_size + 1, self.capacity)

    def sample(self, batch_size):
        indices = random.sample(range(self.memory_size), batch_size)
        state = self.states[indices]
        action = self.actions[indices]
        reward = self.rewards[indices]
        next_state = self.next_states[indices]
        done = self.dones[indices]
        return state, action, reward, next_state, done
    
    def __len__(self):
        return min(self.capacity, self.position)

    def save_buffer(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = "{}/sac_buffer.pkl".format(save_dir)
        print('Saving sac_buffer.pkl to {}'.format(save_path))

        buffer_data = {
            'states': self.states.cpu(),
            'actions': self.actions.cpu(),
            'rewards': self.rewards.cpu(),
            'next_states': self.next_states.cpu(),
            'dones': self.dones.cpu(),
            'position': self.position
        }

        with open(save_path, 'wb') as f:
            pickle.dump(buffer_data, f)
            
        print('Saved sac_buffer.pkl to {}'.format(save_path))

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            buffer_data = pickle.load(f)
            self.states = buffer_data['states'].to(self.device)
            self.actions = buffer_data['actions'].to(self.device)
            self.rewards = buffer_data['rewards'].to(self.device)
            self.next_states = buffer_data['next_states'].to(self.device)
            self.dones = buffer_data['dones'].to(self.device)
            self.position = buffer_data['position']