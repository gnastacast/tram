import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
from glob import glob
from lib.pipeline.tools import video2frames, detect_segment_track


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov', help='input video')
parser.add_argument("--visualization", action='store_true', help='save deva vos for visualization')
parser.add_argument("--low_memory", action='store_true', help='use low-memory models (less accurate)')
args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'{root}/{seq}'
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)
print(f'Running on {file} ...')

##### Extract Frames #####
print('Extracting frames ...')
slam = dict(np.load(f'{seq_folder}/masked_droid_slam.npz'))
rots = slam['suggested_rotation']
print(rots)
# nframes = video2frames(file, img_folder)

import cv2
def video2frames(vidfile, save_folder, start_frame=0, end_frame=None, rots=[]):
    """ Convert input video to images """
    count = 0
    cap = cv2.VideoCapture(vidfile)
    current_rot = None
    while(cap.isOpened()):
        frame_no = count - start_frame + 1
        ret, frame = cap.read()
        if len(rots) and frame_no == int(rots[0][0]):
            current_rot = rots[0][1]
            print(current_rot, frame_no)
            rots = rots[1:]
        if count < start_frame - 1:
            count += 1
            continue
        if end_frame is not None and count > end_frame -1:
            break
        old_size = frame.shape[:2] # old_size is in (height, width) format
        desired_size = max(old_size)
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(frame,
                                    top,
                                    bottom,
                                    left,
                                    right,
                                    cv2.BORDER_CONSTANT,
                                    value=color)


        if ret == True:
            # padded = np.ones((10,10,3), dtype=np.uint8) * 255
            # padded[0:5,0:5,0:2] = 0
            rows,cols = padded.shape[0:2]
            degrees = current_rot / np.pi * 180
            M = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1) 
            padded = cv2.warpAffine(padded, M,(cols,rows)) 
            cv2.imwrite(f'{save_folder}/{frame_no:04d}.jpg', padded)
            count += 1
        else:
            break
    cap.release()
    return count

nframes = video2frames(file, img_folder, 62, 2480, rots)
print(nframes)


##### Detection + SAM + DEVA-Track-Anything #####
print('Detect, Segment, and Track ...')
save_vos = args.visualization
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
low_memory = args.low_memory
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25, 
                                               min_size=100, save_vos=save_vos,
                                               low_memory=low_memory)
np.save(f'{seq_folder}/boxes.npy', boxes_)
np.save(f'{seq_folder}/masks.npy', masks_)
np.save(f'{seq_folder}/tracks.npy', tracks_)

