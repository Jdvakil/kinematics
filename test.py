import kinematics as fk
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
eef_cmd = np.array([ 0.554,  0.2239, 0.9512, 0.0951, -0.9613, -0.2552 , 0.0417 ])
eef_pos = eef_cmd[:3]
eef_quat = eef_cmd[3:7]
print(f"eef_quat = {eef_quat}")
np.set_printoptions(suppress=True)
ik_result = fk.get_ik_action(model_path=model_path,eef_pos=eef_pos, eef_quat=eef_quat)
print(ik_result)
action = ik_result.qpos[:sim.model.nu-1]
print(action)
sim.data.qpos[:7] = action
sim.forward()
viewer.render()
time.sleep(5)
