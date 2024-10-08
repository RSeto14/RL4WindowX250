import mujoco 
import mujoco.viewer
import numpy as np
import os
import sys
import time
import random
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from cfg import Config



class WindowX250Env():
    def __init__(self,headless=True,cfg=Config()):
        self.model = mujoco.MjModel.from_xml_path(parent_dir + r"\trossen_wx250s\scene.xml")
        self.data = mujoco.MjData(self.model)
        self.headless = headless
        self.cfg = cfg
        self.steps = 0
        self.viewer = None
        
    def step(self, action):
        self.control(action)
        self.steps += 1
        
        sensor_data = self.sensor_data()
        observation = np.concatenate([sensor_data])
        reward = self.reward()
        
        termination = False
        truncation = False
        if not self.headless:
            if self.viewer.is_running()==False:
                truncation = True
        info = {}
        
        return observation, reward, termination, truncation, info
        
    
    def sensor_data(self):
        j_names = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"]

        joint_q = np.array([self.data.sensor(j_name + "_pos").data[0] for j_name in j_names])
        joint_dq = np.array([self.data.sensor(j_name + "_vel").data[0] for j_name in j_names])
        
        sensor_data = np.concatenate([joint_q, joint_dq])

        return sensor_data
            
    def control(self,action):
        for i in range(self.cfg.step_dt_per_mujoco_dt): 
            self.data.ctrl[:] = np.concatenate([action])
            mujoco.mj_step(self.model, self.data)
            
        if not self.headless:
            self.viewer.sync()
            if self.data.time > self.steps * self.cfg.mujoco_dt * self.cfg.step_dt_per_mujoco_dt:
                time.sleep(self.data.time - self.steps * self.cfg.mujoco_dt * self.cfg.step_dt_per_mujoco_dt)

    def reset(self):
        self.steps = 0
        self.model = mujoco.MjModel.from_xml_path(parent_dir + r"\trossen_wx250s\scene.xml")
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model,self.data)
        if not self.headless:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.close()
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        return self.sensor_data()
    
    
    
    def reward(self):
        return 0


class ReachingTask(WindowX250Env):
    def __init__(self,headless=True,cfg=Config()):
        super().__init__(headless=headless,cfg=cfg)
        self.target_pos = np.array([0,0,0])
        self.change_target_interval = self.cfg.target_change_interval
        self.next_target_time = self.change_target_interval
        
        self.action_dim = 6
        self.observation_dim = 20
        
    def reset(self):
        sensor_data = super().reset()
        self.next_target_time = self.change_target_interval
        self.target_pos = np.array([random.uniform(self.cfg.target_pos_min[0], self.cfg.target_pos_max[0]), random.uniform(self.cfg.target_pos_min[1], self.cfg.target_pos_max[1]), random.uniform(self.cfg.target_pos_min[2], self.cfg.target_pos_max[2])])
        self.model.body("target_body").pos[:] = self.target_pos
        observation = self.observation()
        return observation
    
    def step(self, action):
        
        action = (np.array(self.cfg.action_space_max) - np.array(self.cfg.action_space_min))/2 * action + (np.array(self.cfg.action_space_max) + np.array(self.cfg.action_space_min))/2
        
        self.control(np.concatenate([action, np.array([0.0])]))
        self.steps += 1
        
        if self.data.time > self.next_target_time:
            self.target_pos = np.array([random.uniform(self.cfg.target_pos_min[0], self.cfg.target_pos_max[0]), random.uniform(self.cfg.target_pos_min[1], self.cfg.target_pos_max[1]), random.uniform(self.cfg.target_pos_min[2], self.cfg.target_pos_max[2])])
            # self.target_pos = np.array([0.5, 0, 0.3])
            self.next_target_time += self.change_target_interval
            self.model.body("target_body").pos[:] = self.target_pos
        
        observation = self.observation()
        reward = self.reward()
        
        termination = False
        truncation = False
        if not self.headless:
            if self.viewer.is_running()==False:
                truncation = True
        if self.steps >= self.cfg.max_steps:
            truncation = True
        info = {}
        
        return observation, reward, termination, truncation, info
    
    def reward(self):
        def f(x):
            return np.exp(-4*(np.abs(x))**2)
        end_effector_pos = self.data.site("end_effector").xpos
        target_pos = self.target_pos
        return f(end_effector_pos[0] - target_pos[0]) + f(end_effector_pos[1] - target_pos[1]) + f(end_effector_pos[2] - target_pos[2])

    def observation(self):
        sensor_data = self.sensor_data()
        target_pos = self.target_pos
        observation = np.concatenate([sensor_data, target_pos, self.data.site("end_effector").xpos])
        return observation
    


if __name__ == "__main__":
    env = ReachingTask(headless=False)
    env.reset()
    done = False
    while not done:
        observation, reward, termination, truncation, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        done = termination or truncation
        print(env.data.time, observation, reward, termination, truncation)
