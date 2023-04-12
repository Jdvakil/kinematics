import numpy as np
from mujoco_py import MjSim
from robohive.utils.quat_math import *
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.physics.sim_scene import SimScene

def forward_kinematics(qpos=np.array([0,0,0,-1.5, 0,0,1.5,0]), model=None):
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
            pos_ee[0]-= 0.1
            pos_ee[-1] -= 0.65
            rot_ee = (np.reshape(sim.data.site_xmat[ee_sid], [3,-1])).T
            rot_ee = (mat2quat(rot_ee))
    return pos_ee, rot_ee


def get_ik_action(model_path=None, eef_pos=np.array([0.55,0,1.2]), eef_quat=np.array([0,1,0,0])):
    ik_sim = SimScene.get_sim(model_path)
    sim = SimScene.get_sim(model_path)
    for i in range(1):

        ik_sim.data.qpos[:7] = np.random.normal(sim.data.qpos[:7], i*0.1)
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
