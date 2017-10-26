# Copyright 2017 Max Planck Society
# Distributed under the BSD-3 Software license,
# (See accompanying file LICENSE.txt or copy at
# https://opensource.org/licenses/BSD-3-Clause)

import numpy as np
import os
import sys
import numba
from PIL import Image
import scipy.io
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import binary_dilation
from cityscapes_data import *

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

def get_reindexed_image(result_labels):
    result_regularIds = np.copy(result_labels)
    for k, v in cIds.iteritems():
        result_regularIds[result_labels==k] = v
    return result_regularIds

def eval_seg_parallel(datatype, gt_label_folder, result_label_folder, imgname, trimap, count):
    max_label = 34
    ignore_label = 255

    tp = np.zeros((max_label))
    fp = np.zeros((max_label))
    fn = np.zeros((max_label))

    edge_tp = np.zeros((max_label))
    edge_fp = np.zeros((max_label))
    edge_fn = np.zeros((max_label))

    print str(count) + ". Image Name: " + imgname
    gt_file = gt_label_folder + imgname + '.png'
    gt_file = os.path.join(gt_label_folder, datatype.lower(), imgname.split('_')[0], imgname + '_gtFine_labelTrainIds.png')
    result_file = result_label_folder + imgname + '.png'

    gt_labels = np.array(Image.open(gt_file))
    gt_labels = get_reindexed_image(gt_labels)
    result_labels = np.array(Image.open(result_file))

    if (np.max(result_labels) > (max_label - 1) and np.max(result_labels)!=255):
        print('Result has invalid labels: ', np.max(result_labels))
    else:
        edge_mask = find_boundaries(gt_labels, connectivity=1)
        edge_mask = binary_dilation(edge_mask, np.ones((trimap, trimap)))
        edge_gt_labels = gt_labels.copy()
        edge_result_labels = result_labels.copy()
        edge_gt_labels[np.equal(edge_mask,0)] = ignore_label
        edge_result_labels[np.equal(edge_mask,0)] = ignore_label
        # For each class
        for class_id in range(0, max_label):
            class_gt = np.equal(gt_labels, class_id)
            class_result = np.equal(result_labels, class_id)
            # import pdb; pdb.set_trace();
            class_result[np.equal(gt_labels, ignore_label)] = 0
            tp[class_id] = tp[class_id] +\
                np.count_nonzero(class_gt & class_result)
            fp[class_id] = fp[class_id] +\
                np.count_nonzero(class_result & ~class_gt)
            fn[class_id] = fn[class_id] +\
                np.count_nonzero(~class_result & class_gt)

            edge_class_gt = np.equal(edge_gt_labels, class_id)
            edge_class_result = np.equal(edge_result_labels, class_id)
            # import pdb; pdb.set_trace();
            edge_class_result[np.equal(edge_gt_labels, ignore_label)] = 0
            edge_tp[class_id] = edge_tp[class_id] +\
                np.count_nonzero(edge_class_gt & edge_class_result)
            edge_fp[class_id] = edge_fp[class_id] +\
                np.count_nonzero(edge_class_result & ~edge_class_gt)
            edge_fn[class_id] = edge_fn[class_id] +\
                np.count_nonzero(~edge_class_result & edge_class_gt)
    return [tp, fp, fn, edge_tp,  edge_fp, edge_fn]

def eval_seg(datatype, result_label_folder, trimap=1, strin=''):
    max_label = 34
    ignore_label = 255

    class_ious = np.zeros((max_label, 1))
    overall_iou = 0
    overall_accuracy = 0

    tp = np.zeros((max_label))
    fp = np.zeros((max_label))
    fn = np.zeros((max_label))

    img_tp = 0
    img_pixels = 0

    edge_class_ious = np.zeros((max_label, 1))
    edge_overall_iou = 0
    edge_overall_accuracy = 0

    edge_tp = np.zeros((max_label))
    edge_fp = np.zeros((max_label))
    edge_fn = np.zeros((max_label))

    edge_img_tp = 0
    edge_img_pixels = 0

    results_ = Parallel(n_jobs=num_cores)(delayed(eval_seg_parallel)(datatype, GT_FOLDER, result_label_folder, VID_NAMES[datatype][i] + '_' + str(GT_FRAMES[datatype][i][0]).zfill(6), trimap, i) for i in range(0,len(VID_NAMES[datatype])))
    [tp, fp, fn, edge_tp, edge_fp, edge_fn] = np.sum(np.array( results_ ),axis=0)

    for class_id in range(0, max_label):
        class_ious[class_id] = tp[class_id] / (tp[class_id] +
                                            fp[class_id] + fn[class_id])
        edge_class_ious[class_id] = edge_tp[class_id] / (edge_tp[class_id] +
                                                edge_fp[class_id] + edge_fn[class_id])
    #import pdb; pdb.set_trace()
    print(result_label_folder)
    print('Overall Class IOUs:' + str(class_ious))
    print('Trimap Class IOU: ' + str(edge_class_ious))
    print('Overall IOU: ' + str(np.nanmean(class_ious)))
    print('Overall Trimap IOU: ' + str(np.nanmean(edge_class_ious)) + ' with trimap: ' + str(trimap))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ' + sys.argv[0] + ' <data_type> <result_label_folder>')
    elif len(sys.argv)==4:
        eval_seg(sys.argv[1], sys.argv[2], trimap=int(sys.argv[3]))
