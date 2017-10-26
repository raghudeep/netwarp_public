// Copyright 2017 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
#include <cfloat>
#include <vector>

#include "caffe/layers/warp_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {

template <typename Dtype>
__global__ void truncate_interp2_fwd(const int nthreads, const Dtype *bottom_0_data_, const Dtype *bottom_1_data_,
                                     const int num_, const int channels_, const int height_, const int width_, 
                                     Dtype *theta_data, Dtype* theta_data_, Dtype *x_w_data, Dtype *top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp = 0;
    const int n = index / (channels_ * height_ * width_);
    temp = index % (channels_ * height_ * width_);   
    const int c = temp / (height_ * width_);
    temp = temp % (height_ * width_);
    const int h = temp / width_;
    const int w = temp % width_;
    int index_x = ((n * 2 + 1) * height_ + h) * width_ + w;
    int index_y = ((n * 2 + 0) * height_ + h) * width_ + w;
    x_w_data[ index_x ] = h + bottom_1_data_[ index_x ];
    x_w_data[ index_y ] = w + bottom_1_data_[ index_y ];
    int xw_floor = (int)floor(x_w_data[ index_x ]);
    int yw_floor = (int)floor(x_w_data[ index_y ]);
    int xw_ceil = (int)ceil(x_w_data[ index_x ]);
    int yw_ceil = (int)ceil(x_w_data[ index_y ]);
    theta_data[ index_x ] = x_w_data[ index_x ] - floor(x_w_data[ index_x ]);
    theta_data[ index_y ] = x_w_data[ index_y ] - floor(x_w_data[ index_y ]);
    theta_data_[ index_x ] = 1 - theta_data[ index_x ];
    theta_data_[ index_y ] = 1 - theta_data[ index_y ];
    int offset = (n * channels_ + c) * height_;
    if (x_w_data[ index_x ] >= 0 && x_w_data[ index_x ] <= height_-1 && 
        x_w_data[ index_y ] >= 0 && x_w_data[ index_y ] <= width_-1) {
      Dtype I0 = bottom_0_data_[ (offset + xw_floor) * width_ + yw_floor ]; 
      Dtype I1 = bottom_0_data_[ (offset + xw_ceil ) * width_ + yw_floor ]; 
      Dtype I2 = bottom_0_data_[ (offset + xw_floor) * width_ + yw_ceil ]; 
      Dtype I3 = bottom_0_data_[ (offset + xw_ceil ) * width_ + yw_ceil ];
      top_data[ (offset +  h) * width_ +  w ] = (theta_data_[index_x] * theta_data_[index_y] * I0) + 
                                                (theta_data[index_x]  * theta_data_[index_y] * I1) + 
                                                (theta_data_[index_x] * theta_data[index_y]  * I2) + 
                                                (theta_data[index_x]  * theta_data[index_y]  * I3);
    }
  }
}

template <typename Dtype>
__global__ void nearest_interp2_fwd(const int nthreads, const Dtype *bottom_0_data_, const Dtype *bottom_1_data_,
                                    const int num_, const int channels_, const int height_, const int width_, 
                                    Dtype *theta_data, Dtype* theta_data_, Dtype *x_w_data, Dtype *top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp = 0;
    const int n = index / (channels_ * height_ * width_);
    temp = index % (channels_ * height_ * width_);   
    const int c = temp / (height_ * width_);
    temp = temp % (height_ * width_);
    const int h = temp / width_;
    const int w = temp % width_;
    int index_x = ((n * 2 + 1) * height_ + h) * width_ + w;
    int index_y = ((n * 2 + 0) * height_ + h) * width_ + w;
    x_w_data[ index_x ] = h + bottom_1_data_[ index_x ];
    x_w_data[ index_y ] = w + bottom_1_data_[ index_y ];
    int xw_floor = (int)floor(x_w_data[ index_x ]);
    int yw_floor = (int)floor(x_w_data[ index_y ]);
    int xw_ceil = (int)ceil(x_w_data[ index_x ]);
    int yw_ceil = (int)ceil(x_w_data[ index_y ]);
    theta_data[ index_x ] = x_w_data[ index_x ] - floor(x_w_data[ index_x ]);
    theta_data[ index_y ] = x_w_data[ index_y ] - floor(x_w_data[ index_y ]);
    if (x_w_data[ index_x ] < 0) {
      theta_data[ index_x ] = x_w_data[ index_x ];
      xw_floor = 0; xw_ceil = 0;
    } 
    if (x_w_data[ index_x ] >= height_-1) {
      theta_data[ index_x ] = x_w_data[ index_x ] - height_;
      xw_floor = height_-1; xw_ceil = height_-1;
    }
    if (x_w_data[ index_y ] < 0) {
      theta_data[ index_y ] = x_w_data[ index_y ];
      yw_floor = 0; yw_ceil = 0;
    }
    if (x_w_data[ index_y ] >= width_-1) {
      theta_data[ index_y ] = x_w_data[ index_y ] - width_;
      yw_floor = width_-1; yw_ceil = width_-1;
    }
    theta_data_[ index_x ] = 1 - theta_data[ index_x ];
    theta_data_[ index_y ] = 1 - theta_data[ index_y ];
    int offset = (n * channels_ + c) * height_;
    Dtype I0 = bottom_0_data_[ (offset + xw_floor) * width_ + yw_floor ]; 
    Dtype I1 = bottom_0_data_[ (offset + xw_ceil ) * width_ + yw_floor ]; 
    Dtype I2 = bottom_0_data_[ (offset + xw_floor) * width_ + yw_ceil ]; 
    Dtype I3 = bottom_0_data_[ (offset + xw_ceil ) * width_ + yw_ceil ];
    top_data[ (offset +  h) * width_ +  w ] = (theta_data_[index_x] * theta_data_[index_y] * I0) + 
                                              (theta_data[index_x]  * theta_data_[index_y] * I1) + 
                                              (theta_data_[index_x] * theta_data[index_y]  * I2) + 
                                              (theta_data[index_x]  * theta_data[index_y]  * I3);
  }
}

template <typename Dtype>
__global__ void truncate_interp2_bwd(const int nthreads, const int num_, const int channels_, const int height_, 
                                     const int width_, const Dtype *theta_data, const Dtype* theta_data_, 
                                     const Dtype *x_w_data, Dtype *bottom_0_diff, Dtype *bottom_1_diff, 
                                     const Dtype *top_diff, const Dtype *top_data, const Dtype *bottom_0_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp = 0;
    const int n = index / (channels_ * height_ * width_);
    temp = index % (channels_ * height_ * width_);   
    const int c = temp / (height_ * width_);
    temp = temp % (height_ * width_);
    const int h = temp / width_;
    const int w = temp % width_;

    int index_x = ((n * 2 + 1) * height_ + h) * width_ + w;
    int index_y = ((n * 2 + 0) * height_ + h) * width_ + w;

    if (!(x_w_data[ index_x ] < 0 || x_w_data[ index_x ] > height_-1 ||
          x_w_data[ index_y ] < 0 || x_w_data[ index_y ] > width_-1)) {
    
      int xw_floor = (int)floor(x_w_data[ index_x ]);
      int yw_floor = (int)floor(x_w_data[ index_y ]);
      int xw_ceil = (int)ceil(x_w_data[ index_x ]);
      int yw_ceil = (int)ceil(x_w_data[ index_y ]);
  
      int bottom_0_index = ((n * channels_ + c) * height_ +  h) * width_ +  w;
      int offset = (n * channels_ + c) * height_;
      Dtype I0 = bottom_0_data[ (offset + xw_floor) * width_ + yw_floor ]; 
      Dtype I1 = bottom_0_data[ (offset + xw_ceil ) * width_ + yw_floor ]; 
      Dtype I2 = bottom_0_data[ (offset + xw_floor) * width_ + yw_ceil ]; 
      Dtype I3 = bottom_0_data[ (offset + xw_ceil ) * width_ + yw_ceil ];
      bottom_1_diff[ index_x ] += ( -1*theta_data_[index_y]*I0 + 
                                       theta_data_[index_y]*I1 - 
                                       theta_data[index_y] *I2 + 
                                       theta_data[index_y] *I3 ) * 
                                  top_diff[(offset + h) * width_ + w];
      bottom_1_diff[ index_y ] += ( -1*theta_data_[index_x]*I0 - 
                                       theta_data[index_x] *I1 + 
                                       theta_data_[index_x]*I2 + 
                                       theta_data[index_x] *I3 ) * 
                                  top_diff[(offset + h) * width_ + w];

      caffe_gpu_atomic_add((Dtype) theta_data_[ index_x ]*theta_data_[ index_y ]*top_diff[bottom_0_index], 
                           bottom_0_diff + ((offset + xw_floor) * width_ + yw_floor ));
      caffe_gpu_atomic_add((Dtype) theta_data[ index_x ] *theta_data_[ index_y ]*top_diff[bottom_0_index],
                           bottom_0_diff + ((offset + xw_ceil ) * width_ + yw_floor ));
      caffe_gpu_atomic_add((Dtype) theta_data_[ index_x ]*theta_data[ index_y ] *top_diff[bottom_0_index],
                           bottom_0_diff + ((offset + xw_floor) * width_ + yw_ceil  ));
      caffe_gpu_atomic_add((Dtype) theta_data[ index_x ] *theta_data[ index_y ] *top_diff[bottom_0_index], 
                           bottom_0_diff + ((offset + xw_ceil ) * width_ + yw_ceil  ));
    }
  }
}
 
template <typename Dtype>
__global__ void nearest_interp2_bwd(const int nthreads, const int num_, const int channels_, const int height_, 
                                    const int width_, const Dtype *theta_data, const Dtype* theta_data_, 
                                    const Dtype *x_w_data, Dtype *bottom_0_diff, Dtype *bottom_1_diff, 
                                    const Dtype *top_diff, const Dtype *top_data, const Dtype *bottom_0_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int temp = 0;
    const int n = index / (channels_ * height_ * width_);
    temp = index % (channels_ * height_ * width_);   
    const int c = temp / (height_ * width_);
    temp = temp % (height_ * width_);
    const int h = temp / width_;
    const int w = temp % width_;

    int index_x = ((n * 2 + 1) * height_ + h) * width_ + w;
    int index_y = ((n * 2 + 0) * height_ + h) * width_ + w;

    int xw_floor = (int)floor(x_w_data[ index_x ]);
    int yw_floor = (int)floor(x_w_data[ index_y ]);
    int xw_ceil = (int)ceil(x_w_data[ index_x ]);
    int yw_ceil = (int)ceil(x_w_data[ index_y ]);

    if (x_w_data[ index_x ] < 0) {
      xw_floor = 0; xw_ceil = 0;
    } 
    if (x_w_data[ index_x ] >= height_-1) {
      xw_floor = height_-1; xw_ceil = height_-1;
    }
    if (x_w_data[ index_y ] < 0) {
      yw_floor = 0; yw_ceil = 0;
    }
    if (x_w_data[ index_y ] >= width_-1) {
      yw_floor = width_-1; yw_ceil = width_-1;
    }

    int bottom_0_index = ((n * channels_ + c) * height_ +  h) * width_ +  w;
    int offset = (n * channels_ + c) * height_;
    Dtype I0 = bottom_0_data[ (offset + xw_floor) * width_ + yw_floor ]; 
    Dtype I1 = bottom_0_data[ (offset + xw_ceil ) * width_ + yw_floor ]; 
    Dtype I2 = bottom_0_data[ (offset + xw_floor) * width_ + yw_ceil ]; 
    Dtype I3 = bottom_0_data[ (offset + xw_ceil ) * width_ + yw_ceil ];
    bottom_1_diff[ index_x ] += ( -1*theta_data_[index_y]*I0 + 
                                     theta_data_[index_y]*I1 - 
                                     theta_data[index_y] *I2 + 
                                     theta_data[index_y] *I3 ) * 
                                top_diff[(offset + h) * width_ + w];
    bottom_1_diff[ index_y ] += ( -1*theta_data_[index_x]*I0 - 
                                     theta_data[index_x] *I1 + 
                                     theta_data_[index_x]*I2 + 
                                     theta_data[index_x] *I3 ) * 
                                top_diff[(offset + h) * width_ + w];
    caffe_gpu_atomic_add((Dtype) theta_data_[ index_x ]*theta_data_[ index_y ]*top_diff[bottom_0_index], 
                         bottom_0_diff + ((offset + xw_floor) * width_ + yw_floor ));
    caffe_gpu_atomic_add((Dtype) theta_data[ index_x ] *theta_data_[ index_y ]*top_diff[bottom_0_index],
                         bottom_0_diff + ((offset + xw_ceil ) * width_ + yw_floor ));
    caffe_gpu_atomic_add((Dtype) theta_data_[ index_x ]*theta_data[ index_y ] *top_diff[bottom_0_index],
                         bottom_0_diff + ((offset + xw_floor) * width_ + yw_ceil  ));
    caffe_gpu_atomic_add((Dtype) theta_data[ index_x ] *theta_data[ index_y ] *top_diff[bottom_0_index], 
                         bottom_0_diff + ((offset + xw_ceil ) * width_ + yw_ceil  ));
  }
}
 

template <typename Dtype>
void WarpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data_0 = bottom[0]->gpu_data(); // image
  const Dtype* bottom_data_1 = bottom[1]->gpu_data(); // optical flow
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* theta_data = theta.mutable_gpu_data();
  Dtype* theta_data_ = theta_.mutable_gpu_data();
  Dtype* x_w_data = x_w.mutable_gpu_data();
  const int num_kernels = num_ * channels_ * height_ * width_;
  caffe_gpu_set(bottom[0]->count(), (Dtype)0., top_data);
  switch (outliers_) {
    case WarpParameter_WarpType_TRUNCATE:
      truncate_interp2_fwd<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
        (num_kernels, bottom_data_0, bottom_data_1, num_, channels_, height_, width_, 
         theta_data, theta_data_, x_w_data, top_data);
      break;
    case WarpParameter_WarpType_NEAREST:
      nearest_interp2_fwd<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
        (num_kernels, bottom_data_0, bottom_data_1, num_, channels_, height_, width_, 
         theta_data, theta_data_, x_w_data, top_data);
      break;

  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || propagate_down[1]) {
    caffe_gpu_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_gpu_diff());
    caffe_gpu_set(bottom[1]->count(), (Dtype)0., bottom[1]->mutable_gpu_diff());
    const Dtype* theta_data = theta.mutable_gpu_data();
    const Dtype* theta_data_ = theta_.mutable_gpu_data();
    const Dtype* x_w_data = x_w.mutable_gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* bottom_0_data = bottom[0]->gpu_data();
    const Dtype* bottom_1_data = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_0_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_1_diff = bottom[1]->mutable_gpu_diff();
    const int num_kernels = num_ * channels_ * height_ * width_;
    switch (outliers_) {
      case WarpParameter_WarpType_NEAREST:
        nearest_interp2_bwd<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
            (num_kernels, num_, channels_, height_, width_, theta_data, theta_data_, x_w_data, 
             bottom_0_diff, bottom_1_diff, top_diff, top_data, bottom_0_data);
        break;
      case WarpParameter_WarpType_TRUNCATE:
        truncate_interp2_bwd<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
            (num_kernels, num_, channels_, height_, width_, theta_data, theta_data_, x_w_data, 
             bottom_0_diff, bottom_1_diff, top_diff, top_data, bottom_0_data);
        break;
    }
    CUDA_POST_KERNEL_CHECK;
    //caffe_gpu_mul(top[0]->count(), top_diff, bottom_0_diff, bottom_0_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WarpLayer);

}  // namespace caffe
