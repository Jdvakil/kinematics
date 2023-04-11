import gym
import robohive
import os
import utils.fock as fk
import h5py
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import time

pos_limits = {'eef_low': np.array([ 0.368 , -0.25  ,  0.9   , -3.1416,  0.    , -3.1416,  0.]), 
              'eef_high': np.array([0.72  , 0.25  , 1.3   , 3.1416, 6.2832, 0.    , 0.835]), 
              'jnt_low': np.array([-2.9, -1, -2.9, -2, -2.9, -1.6573, -2.8973]), 
        'jnt_high': np.array([ 2.8973,  1,  2.8973, 2  ,  2.8973,  1.65,  2.8973])}

model_path = "/mnt/nfs_code/robopen_users/jaydv/utils/robohive/robohive/envs/fm/assets/franka_robotiq.xml"
file_path="/mnt/nfs_code/robopen_users/jaydv/utils/bowl/manga_bowl_20230324-183952.h5"

env_name ='rpFrankaRobotiqData-v0'
env_args = None #{'is_hardware':True}"
env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))

model = load_model_from_path(model_path)
sim = MjSim(model)
viewer = MjViewer(sim)
h5 = h5py.File(file_path, "r")
t = h5[list(h5.keys())[0]]['data']['time'].shape[0]
print(t)
np.set_printoptions(suppress=True)
for trial, value in h5.items():
    if trial == 'Trial0':
        for i in range(t):
            qpos = np.append(value['data/qp_arm'][i], np.array(value['data/qp_ee'][i]))
            print(qpos)
            target_pos, target_quat = fk.forward_kinematics(qpos=qpos[:7], model=model)
            print(target_pos, target_quat)
            eef_cmd = np.concatenate([target_pos, target_quat])
            eef_pos = eef_cmd[:3]
            eef_quat = eef_cmd[3:7]
            ik_result = fk.get_ik_action(model_path=model_path,eef_pos=eef_pos, eef_quat=eef_cmd)
            print(ik_result.success)
            action = ik_result.qpos[:sim.model.nu-1]
            # action = 2*(((action - pos_limits['jnt_low'])/(pos_limits['jnt_high']-pos_limits['jnt_low']))-0.75)
            print(action)
            sim.data.qpos[:7] = action
            sim.forward()
            time.sleep(0.05)
            viewer.render()

            



'''
import utils.fock as fk
import numpy as np
from  mujoco_py import load_model_from_path, MjSim, MjViewer
import time

pos_limits = {'eef_low': np.array([ 0.368 , -0.25  ,  0.9   , -3.1416,  0.    , -3.1416,  0.]), 
              'eef_high': np.array([0.72  , 0.25  , 1.3   , 3.1416, 6.2832, 0.    , 0.835]), 
              'jnt_low': np.array([-2.9, -1, -2.9, -2, -2.9, -1.6573, -2.8973]), 
        'jnt_high': np.array([ 2.8973,  1,  2.8973, 2  ,  2.8973,  1.65,  2.8973])}

model_path = "/mnt/nfs_code/robopen_users/jaydv/utils/robohive/robohive/envs/fm/assets/franka_robotiq.xml"
model = load_model_from_path(model_path)
sim = MjSim(model)
viewer = MjViewer(sim)
# eef_cmd = np.array([ 0.5738, -0.1328,  0.9802, 0.0888 ,-0.9924,  0.0776,  0.0363])
eef_cmd = np.array([ 0.5855, -0.0716 , 1.2098,0.0479, -0.9976,  0.0221, -0.0455])
eef_pos = eef_cmd[:3]
eef_quat = eef_cmd[3:7]
print(f"eef_quat = {eef_quat}")
np.set_printoptions(suppress=True)
ik_result = fk.get_ik_action(model_path=model_path,eef_pos=eef_pos, eef_quat=eef_quat)
print(ik_result)
action = ik_result.qpos[:sim.model.nu-1]
#action = 2*(((action - pos_limits['jnt_low'])/(pos_limits['jnt_high']-pos_limits['jnt_low']))-0.75)
print(action)
sim.data.qpos[:7] = action
sim.forward()
viewer.render()
time.sleep(5)
'''
