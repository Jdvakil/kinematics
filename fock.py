import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from robohive.physics.sim_scene import SimScene
from robohive.utils.quat_math import *
from robohive.utils.inverse_kinematics import qpos_from_site_pose
import gym
def forward_kinematics(qpos=None, model=None):
    sim = MjSim(model)
    ee_sid = sim.model.site_name2id("end_effector")
    pos_ee = None
    rot_ee = None
    for i in range(1):
        if qpos is not None:
            sim.data.qpos[:7] = np.array(qpos, dtype=float)
            sim.forward()
            sim.data.time += sim.model.opt.timestep
            pos_ee = sim.data.site_xpos[ee_sid]
            rot_ee = (np.reshape(sim.data.site_xmat[ee_sid], [3,3])).transpose()
            rot_ee = (mat2quat(rot_ee))
    return pos_ee, rot_ee

def get_ik_action(model_path, eef_pos, eef_quat):
    ik_sim = SimScene.get_sim(model_path)
    sim = SimScene.get_sim(model_path)
    for i in range(1):

        ik_sim.data.qpos[:7] = np.random.normal(sim.data.qpos[:7], i*0.1)

        # ik_sim.data.qpos[2] = 0.0
        ik_sim.data.qpos[3] = -2.0
        ik_sim.forward()

        ik_result = qpos_from_site_pose(physics = ik_sim,
                                        site_name = "end_effector",
                                        target_pos= eef_pos,
                                        target_quat= eef_quat,
                                        inplace=False,
                                        regularization_strength=1.0)

        if ik_result.success:
            break
    return ik_result
# import robohive
# import gym
# import numpy as np
# import fock
# from mujoco_py import load_model_from_path, MjSim, MjViewer
# from robohive.physics.sim_scene import SimScene
# from robohive.utils.quat_math import *
# from robohive.utils.inverse_kinematics import qpos_from_site_pose
# import gym

# env_name ='rpFrankaRobotiqData-v0'
# env_args = "{'is_hardware':True}"
# env = gym.make(env_name) if env_args==None else gym.make(env_name, **(eval(env_args)))
# env.reset(blocking=True)

# home_polymetis = np.array([ 0, 0, 0, -1.5, 0, 1.5,  0])
# home_qpos = np.concatenate([home_polymetis, np.array([0])])
# print("moving to home_qpos  -- polymetis")
# env.step(home_qpos)
# sim_path = "/mnt/nfs_code/robopen_users/jaydv/utils/robohive/robohive/envs/fm/assets/franka_robotiq.xml"
# model = load_model_from_path(sim_path)
# target_pos, target_quat = fock.forward_kinematics(qpos=home_polymetis, model=model)
# target_pos[0]-=0.10
# target_pos[-1] -= 1
# print(target_pos, target_quat)

# sim = SimScene.get_sim(model_handle=sim_path)
# ik_result = qpos_from_site_pose(
#                 physics = sim,
#                 site_name = "end_effector",
#                 target_pos= target_pos,
#                 target_quat= target_quat,
#                 inplace=False,
#                 regularization_strength=10)
# qpos = ik_result.qpos[:7]
# print(qpos)


# # sim_path = "/mnt/nfs_code/robopen_users/jaydv/utils/robohive/robohive/envs/fm/assets/franka_robotiq.xml"
# # model = load_model_from_path(sim_path)
# # target_pos, target_rot = forward_kinematics(qpos=home_pose, model=model)
# # print(target_pos, target_rot)
# # sim = SimScene.get_sim(model_handle=sim_path)
# # ik_result = qpos_from_site_pose(
# #                 physics = sim,
# #                 site_name = "end_effector",
# #                 target_pos= target_pos,
# #                 target_quat= target_rot,
# #                 inplace=False,
# #                 regularization_strength=1.0)
# # qpos = ik_result.qpos[:7]
# # print(qpos)

