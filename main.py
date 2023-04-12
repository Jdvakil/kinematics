import kinematics 
import numpy as np
import time
from  mujoco_py import load_model_from_path, MjSim, MjViewer
import h5py
import click

pos_limits = {'eef_low': np.array([ 0.368 , -0.25  ,  0.9   , -3.1416,  0.    , -3.1416,  0.]), 
              'eef_high': np.array([0.72  , 0.25  , 1.3   , 3.1416, 6.2832, 0.    , 0.835]), 
              'jnt_low': np.array([-2.9, -1, -2.9, -2, -2.9, -1.6573, -2.8973]), 
        'jnt_high': np.array([ 2.8973,  1,  2.8973, 2  ,  2.8973,  1.65,  2.8973])}

@click.command(help="")
@click.option("-m", '--model_path', required=True, type=str, help="Init the sim")
@click.option('-rp', '--rollout_path', required=True, type=str,help="rollout path for the h5")
def main(model_path, rollout_path):
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    qpos = None
    h5 = h5py.File(rollout_path, "r") 
    hor = h5[list(h5.keys())[0]]['data']['time'].shape[0]
    np.set_printoptions(suppress=True)
    for trial, value in h5.items():
        if trial == 'Trial0':
            for i in range(hor):
                qpos = value['data/qp_arm'][i][:7]
                print("raw qpos -- ", qpos)
                pos,rot = kinematics.forward_kinematics(qpos=qpos, model=model)
                pos[-1] += 0.65
                eef_cmd = np.concatenate([pos, rot])
                print("pos, rot --", eef_cmd)
                eef_pos = eef_cmd[:3]
                eef_quat = eef_cmd[3:7]
                ik_result = kinematics.get_ik_action(model_path=model_path,eef_pos=eef_pos, eef_quat=eef_quat)
                print("ik_result -- ",ik_result)
                action = ik_result.qpos[:sim.model.nu-1]
                sim.data.qpos[:7] = action
                sim.forward()
                viewer.render()
                time.sleep(0.05)

if __name__ == '__main__':
    main()