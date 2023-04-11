import pickle
import os
import torch
from natsort import natsorted, ns
abs_dir = '/mnt/nfs_code/robopen_users/jaydv/utils' 
task = 'bowl/pos'
file_path = os.path.join(abs_dir, task)
for i in range(1):
    path = (os.path.join(file_path, str(i)))
    for r,d,f in os.walk(path):
        f = natsorted(f, alg=ns.IGNORECASE)
        for files in f:
            files = files.replace(".pickle", "")
            if int(files.split("_")[-1]) % 3 == 0:
                ee_pose  = (files+".pickle")
                ee_pose_path = os.path.join(path, ee_pose)
                with open(ee_pose_path, 'rb+') as f:
                    pose = torch.Tensor((pickle.load(f)))
                    pose[-1] += 0.7
                    print(pose)
