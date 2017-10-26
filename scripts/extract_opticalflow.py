# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

# extract and save optical flow
# python extract_opticalflow.py VAL
# python extract_opticalflow.py TEST

import os
import sys
sys.path.insert(0,'.')
import numpy as np
from cityscapes_data import *

num_prev_frames = 1 # 7
key = sys.argv[1] # options : 'DEMO', 'TEST', 'VAL', 'TRAIN'

print "Total # Videos = " + str(len(VID_NAMES[key])) + "\n"
for i in range(0, len(VID_NAMES[key])):
    vid_name  = VID_NAMES[key][i]
    [city_name, _] = vid_name.split('_')
    print "Extracting optical flow for video " + str(i+1) + "/" + str(len(VID_NAMES[key]))
    for gt_frame  in GT_FRAMES[key][i]:
        for j in range(0,num_prev_frames+1):
            curr_indx = gt_frame - j
            prev_indx = gt_frame - j - 1
            curr_frame_path = ('%s/leftImg8bit_sequence/%s/%s/%s_%06d_leftImg8bit.png ' % (os.environ['CITYSCAPES_DATASET'], key.lower(), city_name, vid_name, curr_indx))
            prev_frame_path = ('%s/leftImg8bit_sequence/%s/%s/%s_%06d_leftImg8bit.png ' % (os.environ['CITYSCAPES_DATASET'], key.lower(), city_name, vid_name, prev_indx))
            flo_path = ('%s/dis_flow/%s/%s/%s_%06d_leftImg8bit.flo' % (os.environ['CITYSCAPES_DATASET'], key.lower(), city_name, vid_name, curr_indx))
            if not os.path.exists(os.path.dirname(flo_path)):
                os.makedirs(os.path.dirname(flo_path))
            cmd = os.environ['NETWARP_BUILD_DIR'] + '/external/OF_DIS/run_OF_RGB ' + curr_frame_path + ' ' + prev_frame_path + ' ' + flo_path
            os.system(cmd)
