import mujoco 
import mujoco.viewer
import numpy as np
import os
import sys
import time
import random
import pytz
import datetime
import imageio
# import cv2

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from cfg import Config



class WindowX250Env():
    def __init__(self,headless=True,cfg=Config(),capture=False):
        self.model = mujoco.MjModel.from_xml_path(parent_dir + r"\trossen_wx250s\scene.xml")
        self.data = mujoco.MjData(self.model)
        self.headless = headless
        self.cfg = cfg
        self.steps = 0
        self.viewer = None
        self.capture = capture
        self.video_height = cfg.video_height
        self.video_width = cfg.video_width
        self.video_fps = cfg.video_fps
        self.frames = []
        
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
        joint_frc = np.array([self.data.sensor(j_name + "_frc").data[0] for j_name in j_names])

        return joint_q, joint_dq, joint_frc
            
    def control(self,action):
        for i in range(self.cfg.step_dt_per_mujoco_dt): 
            self.data.ctrl[:] = np.concatenate([action])
            mujoco.mj_step(self.model, self.data)
            
        if not self.headless:
            self.viewer.sync()
            if self.data.time > self.steps * self.cfg.mujoco_dt * self.cfg.step_dt_per_mujoco_dt:
                time.sleep(self.data.time - self.steps * self.cfg.mujoco_dt * self.cfg.step_dt_per_mujoco_dt)
                
        if self.capture == True:
            if len(self.frames) < self.data.time * self.video_fps:
                self.renderer.update_scene(self.data,camera="cam1")
                pixels = self.renderer.render()
                self.frames.append(pixels)
                
    def save_video(self,video_folder=f"{script_dir}/Videos", video_name=None):
        if self.capture == True:
            if len(self.frames) > 0:
                
                timezone = pytz.timezone('Asia/Tokyo')
                current_datetime = datetime.datetime.now(timezone)
                formatted_datetime = current_datetime.strftime("%y%m%d_%H%M%S")
                
                
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                
                if video_name == None:
                    video_path = video_folder + f"/{formatted_datetime}.mp4"
                else:
                    video_path = video_folder + f"/{video_name}"
                
                
                imageio.mimsave(video_path, self.frames, fps=self.video_fps, macro_block_size=1)
                # # Define the codec and create VideoWriter object
                # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
                # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V') 
                # out = cv2.VideoWriter(video_path, fourcc, self.video_fps, (self.video_width, self.video_height))
                
                # for frame in self.frames:
                #     # Convert RGB to BGR (OpenCV uses BGR format)
                #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #     out.write(frame_bgr)
                
                # out.release()
                self.frames = []
                print(f"Saved video to {video_path}")


    def reset(self,video_folder=f"{script_dir}/Videos"):
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
                
        if self.capture == True:
            self.save_video(video_folder=video_folder)
            self.renderer = mujoco.Renderer(self.model,self.video_height,self.video_width)


    
    
    def reward(self):
        return 0


class ReachingTask(WindowX250Env):
    def __init__(self,headless=True,cfg=Config(),capture=False):
        super().__init__(headless=headless,cfg=cfg,capture=capture)
        self.target_pos = np.array([0,0,0])
        self.change_target_interval = self.cfg.target_change_interval
        self.next_target_time = self.change_target_interval
        
        self.action_dim = 6
        self.observation_dim = 20
        
    def reset(self,video_folder=f"{script_dir}/Videos"):
        super().reset(video_folder=video_folder)
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
        
        end_effector_pos = self.data.site("end_effector").xpos
        target_pos = self.target_pos

        
        rew_pos = np.linalg.norm(end_effector_pos - target_pos)
        
        joint_q, joint_dq, joint_frc = self.sensor_data()
        
        rew_vel = np.sum(np.abs(joint_dq))
        rew_frc = np.sum(np.abs(joint_frc))
        
        return self.cfg.reward_pos * rew_pos + self.cfg.reward_vel * rew_vel + self.cfg.reward_frc * rew_frc

    def observation(self):
        joint_q, joint_dq, joint_frc = self.sensor_data()
        target_pos = self.target_pos
        observation = np.concatenate([joint_q, joint_dq, target_pos, self.data.site("end_effector").xpos])
        return observation
    


if __name__ == "__main__":
    env = ReachingTask(headless=False)
    env.reset()
    done = False
    while not done:
        observation, reward, termination, truncation, info = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        done = termination or truncation
        print(env.data.time, observation, reward, termination, truncation)
