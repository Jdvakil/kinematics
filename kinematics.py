import numpy as np
from mujoco_py import MjSim
from robohive.utils.quat_math import *
from robohive.utils.inverse_kinematics import qpos_from_site_pose

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
            pos_ee[0]-= 0.1
            pos_ee[-1] -= 0.65
            rot_ee = (np.reshape(sim.data.site_xmat[ee_sid], [3,-1])).T
            # rot_ee[[0, -1]] = rot_ee[[-1, 0]]
            # rot_ee[1] *= -1
            rot_ee = (mat2quat(rot_ee))
    return pos_ee, rot_ee

def inverse_kinematics(sim=None, eef_pos=None, eef_quat=None):
    for i in range(1):
        sim.data.qpos[:7] = np.random.normal(sim.data.qpos[:7], i*0.1)
        sim.forward()

        ik_result = qpos_from_site_pose(physics = sim,
                                        site_name = "end_effector",
                                        target_pos= eef_pos,
                                        target_quat= eef_quat,
                                        inplace=False,
                                        regularization_strength=1.0)

        if ik_result.success:
            break
    return ik_result
