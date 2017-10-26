# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

import os
import numpy as np
import cv2
from PIL import Image
from cityscapes_data import *
import cv2

CROP_SIZE = 713

def fetch_and_transform_image(image_path, scale_factor=1.0):
    """fetch and mirror an image with a fixed label_margin"""
    img = np.array(Image.open(image_path)).astype(np.float32) - mean_pixel
    img = img[:,:,[2,1,0]]
    height_orig, width_orig = img.shape[:2]
    img = cv2.resize(img, (int(np.round(scale_factor*width_orig)+1), int(np.round(scale_factor*height_orig)+1)))
    height, width = img.shape[:2]
    img = np.expand_dims(img.transpose([2, 0, 1]), axis=0)
    if height < CROP_SIZE:
        img = np.lib.pad(img, ((0, 0), (0, 0), (0, CROP_SIZE - height), (0, 0)), 'constant', constant_values=0)
    if width < CROP_SIZE:
        img = np.lib.pad(img, ((0, 0), (0, 0), (0, 0), (0, CROP_SIZE - width)), 'constant', constant_values=0)
    return [img, height, width]

def fetch_flo(flo_path, scale_factor=1.0):
    with open(flo_path, mode='rb') as f:
        ftag = np.fromfile(f,dtype=np.float32,count=1)
        if ftag!=202021.25:
            raise('Error in reading flo')
        width = np.fromfile(f,dtype=np.int32,count=1)
        height = np.fromfile(f,dtype=np.int32,count=1)
        flo = np.fromfile(f,dtype=np.float32,count=-1)
    flo = np.reshape(flo,(height,width,2))
    flo = scale_factor * cv2.resize(flo, (int(np.round(scale_factor*width)+1), int(np.round(scale_factor*height)+1)))
    flo = np.expand_dims(flo.transpose([2,0,1]), axis=0)
    height, width = flo.shape[2:]
    if height < CROP_SIZE:
        flo = np.lib.pad(flo, ((0, 0), (0, 0), (0, CROP_SIZE - height), (0, 0)), 'constant', constant_values=0)
    if width < CROP_SIZE:
        flo = np.lib.pad(flo, ((0, 0), (0, 0), (0, 0), (0, CROP_SIZE - width)), 'constant', constant_values=0)
    #print(np.min(flo))
    #print(np.max(flo))
    return flo

def get_scaled_img_flo_array(data_type, seq_id, frame_idx, scale_factor, num_prev_frames):
    img_array, flo_array = [[], []]
    for i in range(num_prev_frames  ,-1,-1):
        if frame_idx - i >= 0:
            cityname = VID_NAMES[data_type][seq_id].split('_')[0]
            imgname = VID_NAMES[data_type][seq_id] + '_' + str(frame_idx - i).zfill(6) + '_leftImg8bit.png'
            [img, height, width] = fetch_and_transform_image(os.path.join(IMG_FOLDER, data_type.lower().split('_')[0], cityname, imgname),
                                            scale_factor=scale_factor)
            img_array.append(img)
    for i in range(num_prev_frames-1,-1,-1):
       if frame_idx - i >= 0:
           cityname = VID_NAMES[data_type][seq_id].split('_')[0]
           imgname = VID_NAMES[data_type][seq_id] + '_' + str(frame_idx - i).zfill(6) + '_leftImg8bit.flo'
           flo = fetch_flo(os.path.join(FLO_FOLDER, data_type.lower().split('_')[0], cityname, imgname),
                                        scale_factor=scale_factor)
           flo_array.append(flo)
    return [img_array, flo_array, height, width]
    #import pdb; pdb.set_trace()
