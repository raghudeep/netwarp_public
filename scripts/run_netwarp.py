# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

from __future__ import print_function, division
import os
import sys
cityscapesscripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../external/cityscapesScripts/cityscapesscripts/evaluation/')
sys.path.insert(0,cityscapesscripts_dir)

from cityscapes_data import *
from fetch_and_transform_data import *
import evalPixelLevelSemanticLabeling

import cv2
import numpy as np
import time
from PIL import Image

from joblib import Parallel, delayed
import multiprocessing

SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
STRIDE = 476

def softmax(unaries):
    """softmax operation on a given matrix"""
    maxes = np.amax(unaries, axis=0)
    maxes = maxes.reshape(1, maxes.shape[0], maxes.shape[1])
    ex = np.exp(unaries - maxes)
    dist = ex / np.sum(ex, axis=0)
    return dist

def get_colored_image(result_labels):
    result_image = Image.fromarray(np.uint8(result_labels))
    result_image.putpalette(palette)
    return result_image

def get_reindexed_image(result_labels):
    result_regularIds = np.copy(result_labels)
    for k, v in cIds.iteritems():
        result_regularIds[result_labels==k] = v
    result_image = Image.fromarray(result_regularIds.astype(np.uint8))
    return result_image

def predict_in_parallel(datatype, prototxt, caffemodel, out_dir, num_gpus, gpu_indx):
    caffe_root = os.path.join(os.environ['NETWARP_BUILD_DIR'], 'tmp_caffe_clone/src/CaffeUpstream/')
    sys.path.insert(0, caffe_root + 'python')
    print('Using CAFFE from ' + caffe_root + ' on GPU ' + str(gpu_indx))
    import caffe
    caffe.set_mode_gpu()
    caffe.set_device(gpu_indx)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('Loaded ' + prototxt + ' with ' + caffemodel + ' for GPU ' + str(gpu_indx))

    index_out_dir = os.path.join(out_dir, 'index/')
    color_out_dir = os.path.join(out_dir, 'color/')

    img_indx = range(0, len(VID_NAMES[datatype]))
    gpu_img_indx = [img_indx[i::num_gpus] for i in xrange(num_gpus)][gpu_indx]
    for vid_num in gpu_img_indx:
        pres_vid_name = VID_NAMES[datatype][vid_num]
        for frame_idx in GT_FRAMES[datatype][vid_num]:
            frame_name = pres_vid_name + '_' + str(frame_idx).zfill(6)
            print(str(vid_num) + '. ' + frame_name + ' ' + time.strftime('%l:%M:%S%p %Z on %b %d, %Y') + ' on GPU ' + str(gpu_indx))
            index_output_path = index_out_dir + '/' + frame_name + '.png'
            color_output_path = color_out_dir + '/' + frame_name + '.png'
            if not os.path.isfile(index_output_path):
                prob_all = np.zeros((19,1024,2048))

                for scale_factor in SCALES:
                    [img_array, flo_array, height_resize, width_resize] = get_scaled_img_flo_array(datatype, vid_num, frame_idx, scale_factor, 2)
                    height, width = img_array[0].shape[2:]
                    h_grid = int(np.ceil((height-CROP_SIZE)/STRIDE) + 1)
                    w_grid = int(np.ceil((width-CROP_SIZE)/STRIDE) + 1)
                    prob_scale = np.zeros((1,19,height,width))
                    cnt_scale = np.zeros((1,19,height,width))

                    for grid_yidx in range(0,h_grid):
                        for grid_xidx in range(0,w_grid):
                            s_x = (grid_xidx) * STRIDE + 1
                            s_y = (grid_yidx) * STRIDE + 1
                            e_x = np.min([s_x + CROP_SIZE - 1, width])
                            e_y = np.min([s_y + CROP_SIZE - 1, height])
                            s_x = e_x - CROP_SIZE + 1 -1
                            s_y = e_y - CROP_SIZE + 1 -1

                            inputs = {}
                            inputs['conv5_4_1'] = np.zeros((1,512,90,90))
                            inputs['flo_1'] = np.zeros((1,2,713,713))
                            inputs['data_1'] = np.zeros((1,3,713,713))
                            inputs['data_0'] = img_array[0][:,:,s_y:e_y,s_x:e_x]

                            for j in range(0,2):
                                out = net.forward_all(**inputs)
                                inputs = {}
                                if j + 1 < len(img_array):
                                    inputs['conv5_4_1'] = net.blobs['conv5_4'].data[:,:,:,:]
                                    inputs['flo_1'] = flo_array[j][:,:,s_y:e_y,s_x:e_x]
                                    inputs['data_0'] = img_array[j+1][:,:,s_y:e_y,s_x:e_x]
                                    inputs['data_1'] = img_array[j][:,:,s_y:e_y,s_x:e_x]
                            score = net.blobs['upsampled'].data[0,:,:,:].copy()

                            inputs = {} # flip
                            inputs['conv5_4_1'] = np.zeros((1,512,90,90))
                            inputs['flo_1'] = np.zeros((1,2,713,713))
                            inputs['data_1'] = np.zeros((1,3,713,713))
                            inputs['data_0'] = img_array[0][:,:,s_y:e_y,s_x:e_x]
                            inputs['data_0'] = inputs['data_0'][:,:,:,::-1]

                            for j in range(0,2):
                                out = net.forward_all(**inputs)
                                inputs = {}
                                if j + 1 < len(img_array):
                                    inputs['conv5_4_1'] = net.blobs['conv5_4'].data[:,:,:,:]
                                    inputs['flo_1'] = -1 * flo_array[j][:,:,s_y:e_y,s_x:e_x]
                                    inputs['data_0'] = img_array[j+1][:,:,s_y:e_y,s_x:e_x]
                                    inputs['data_1'] = img_array[j][:,:,s_y:e_y,s_x:e_x]
                                    inputs['data_0'] = inputs['data_0'][:,:,:,::-1]
                                    inputs['data_1'] = inputs['data_1'][:,:,:,::-1]
                            score_flip = net.blobs['upsampled'].data[0,:,:,::-1].copy()

                            score = softmax(score + score_flip)
                            cnt_scale[:,:,s_y:e_y,s_x:e_x] = cnt_scale[:,:,s_y:e_y,s_x:e_x] + 1
                            prob_scale[:,:,s_y:e_y,s_x:e_x] = prob_scale[:,:,s_y:e_y,s_x:e_x] + score

                    prob_scale = prob_scale / cnt_scale
                    prob_scale = prob_scale[0,:,:height_resize,:width_resize]
                    prob_scale = prob_scale.transpose([1,2,0])
                    prob_scale = cv2.resize(prob_scale, (2048,1024))
                    prob_all = prob_all + prob_scale.transpose([2,0,1])

                prediction = np.argmax(prob_all.transpose([1, 2, 0]), axis=2)

                seg_map = get_reindexed_image(prediction)
                if not os.path.exists(os.path.dirname(index_output_path)):
                    os.makedirs(os.path.dirname(index_output_path))
                seg_map.save(index_output_path)

                seg_map = get_colored_image(prediction)
                if not os.path.exists(os.path.dirname(color_output_path)):
                    os.makedirs(os.path.dirname(color_output_path))
                seg_map.save(color_output_path)


def predict(datatype, prototxt, caffemodel, out_dir, ngpus):

    index_out_dir = os.path.join(out_dir, 'index/')
    color_out_dir = os.path.join(out_dir, 'color/')
    if not os.path.exists(index_out_dir):
        os.makedirs(index_out_dir)
    if not os.path.exists(color_out_dir):
        os.makedirs(color_out_dir)

    Parallel(n_jobs=ngpus)(delayed(predict_in_parallel)(datatype, prototxt, caffemodel, out_dir, ngpus, i)
                               for i in range(ngpus))

    os.environ["CITYSCAPES_RESULTS"] = index_out_dir
    evalPixelLevelSemanticLabeling.main([])

if __name__ == '__main__':
    if len(sys.argv)==6:
        predict(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]))
    else:
        print('python run_netwarp.py VAL path_to_prototxt path_to_caffemodel results_dir num_gpus')

