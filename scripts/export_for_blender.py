import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
from glob import glob
import numpy as np
import os
import torch
from smplx import SMPL

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov')
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

##### Read results from SLAM and HPS #####
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'{root}/{seq}'
img_folder = f'{seq_folder}/images'
hps_folder = f'{seq_folder}/hps'
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
hps_files = sorted(glob(f'{hps_folder}/*.npy'))
# hps_output_folder = f'{seq_folder}/hps_numpy'
# os.makedirs(hps_output_folder, exist_ok=True)

# kwargs = {}
# kwargs["model_path"] = "data/smpl"

# smpl = SMPL(create_body_pose=True,
#             create_betas=True,
#             create_global_orient=True,
#             create_transl=True,
#             **kwargs)
# colors = np.loadtxt('data/colors.txt')/255

# max_track = len(hps_files)

# for i in range(max_track):
#     hps_file = hps_files[i]
#     data = np.load(hps_file, allow_pickle=True).item()
#     data_np = {}
#     for key, val in data.items():
#         data_np[key] = val.numpy()
#     np.save(os.path.join(hps_output_folder,
#             os.path.basename(hps_file)),
#             data_np)

# base_output = smpl.forward()
# npz_data = {
#     'betas': base_output.betas.detach().cpu().numpy(),
#     'gender': smpl.gender,
#     'pose': base_output.body_pose.detach().cpu().numpy(),
#     'trans': base_output.transl,
# }

# print("\n===============\nmasked_droid_slam.npz\n")

slam = {}

for key, value in np.load(f"{seq_folder}/masked_droid_slam.npz").items():
    if key == "traj":
        print(f"{key}: ({value.dtype}, {value.shape})")
    else:
        print(f"{key}: {value}")
    slam[key] = value

slam['img_focal'] = 1986
slam['img_center'] = [715, 715]

np.savez(f"{seq_folder}/masked_droid_slam.npz", ** slam, allow_pickle=False)


np.savez(os.path.join(seq_folder, "smpl_avatar.npz"), **npz_data, allow_pickle=False)

print("\n===============\nsmpl_avatar.npz\n")

for key, value in npz_data.items():
    if type(value) == np.ndarray:
        print(f"{key}: ({value.dtype}, {value.shape})")
    else:
        print(f"{key}: {value}")

print(f"\n===============\nhps_numpy/{hps_file}\n")

for key, value in data_np.items():
    if type(value) == np.ndarray:
        print(f"{key}: ({value.dtype}, {value.shape})")
    else:
        print(f"{key}: {value}")