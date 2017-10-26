// Copyright 2017 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
#include <cfloat>
#include <vector>
#include <math.h>

#include "caffe/layers/warp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WarpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(top.size(), 1);
  vector<int> bottom_0_shape = bottom[0]->shape();
  vector<int> bottom_1_shape = bottom[1]->shape();
  CHECK_EQ(bottom_0_shape[2], bottom_1_shape[2])
    << "Optical Flow dimensions need to match input blob.";
  CHECK_EQ(bottom_0_shape[3], bottom_1_shape[3])
    << "Optical Flow dimensions need to match input blob.";
  outliers_ = this->layer_param_.warp_param().outliers();
}

template <typename Dtype>
void WarpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> bottom_0_shape = bottom[0]->shape();
  vector<int> bottom_1_shape = bottom[1]->shape();
  top[0]->Reshape(bottom_0_shape);
  theta.Reshape(bottom_1_shape);
  theta_.Reshape(bottom_1_shape);
  x_w.Reshape(bottom_1_shape);
  num_ = bottom_0_shape[0];
  channels_ = bottom_0_shape[1];
  height_ = bottom_0_shape[2];
  width_ = bottom_0_shape[3];
}

template <typename Dtype>
void WarpLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_0_data_ = bottom[0]->cpu_data();
  const Dtype* bottom_1_data_ = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* theta_data = theta.mutable_cpu_data();
  Dtype* theta_data_ = theta_.mutable_cpu_data();
  Dtype* x_w_data = x_w.mutable_cpu_data();
  caffe_set(bottom[0]->count(), (Dtype)0., top_data);
  caffe_set(bottom[1]->count(), (Dtype)0., theta_data);
  caffe_set(bottom[1]->count(), (Dtype)0., theta_data_);
  caffe_set(bottom[1]->count(), (Dtype)0., x_w_data);
  for (int n=0; n<num_; n++) {
    for (int c=0; c<channels_; c++) {
      for (int h=0; h<height_; h++) {
        for (int w=0; w<width_; w++) {
          int index_x = ((n * 2 + 1) * height_ + h) * width_ + w;
          int index_y = ((n * 2 + 0) * height_ + h) * width_ + w;
          x_w_data[ index_x ] = h + bottom_1_data_[ index_x ];// + 0.00000001;
          x_w_data[ index_y ] = w + bottom_1_data_[ index_y ];// + 0.00000001;
          int xw_floor = (int)floor(x_w_data[ index_x ]);
          int yw_floor = (int)floor(x_w_data[ index_y ]);
          int xw_ceil = (int)ceil(x_w_data[ index_x ]);
          int yw_ceil = (int)ceil(x_w_data[ index_y ]);
          theta_data[ index_x ] = x_w_data[ index_x ] - floor(x_w_data[ index_x ]);
          theta_data[ index_y ] = x_w_data[ index_y ] - floor(x_w_data[ index_y ]);
          if (outliers_ == WarpParameter_WarpType_NEAREST) {
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
          }
          theta_data_[ index_x ] = 1 - theta_data[ index_x ];
          theta_data_[ index_y ] = 1 - theta_data[ index_y ];
          int offset = (n * channels_ + c) * height_;

          if (!(outliers_ == WarpParameter_WarpType_TRUNCATE && 
                (x_w_data[ index_x ] < 0 || 
                 x_w_data[ index_x ] > height_-1 || 
                 x_w_data[ index_y ] < 0 || 
                 x_w_data[ index_y ] > width_-1))) {
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
    }
  }
}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0] || propagate_down[1]) {
    caffe_set(bottom[0]->count(), (Dtype)0., bottom[0]->mutable_cpu_diff());
    caffe_set(bottom[1]->count(), (Dtype)0., bottom[1]->mutable_cpu_diff());
  
    const Dtype* theta_data = theta.cpu_data();
    const Dtype* theta_data_ = theta_.cpu_data();
    const Dtype* x_w_data = x_w.cpu_data();
  
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_0_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_1_diff = bottom[1]->mutable_cpu_diff();
  
    //const Dtype* top_data = top[0]->cpu_data();
    const Dtype* bottom_0_data = bottom[0]->cpu_data();
    //const Dtype* bottom_1_data = bottom[1]->cpu_data();
  
    for (int n=0; n<num_; n++) {
      for (int h=0; h<height_; h++) {
        for (int w=0; w<width_; w++) {

          int index_x = ((n * 2 + 1) * height_ + h) * width_ + w;
          int index_y = ((n * 2 + 0) * height_ + h) * width_ + w;
  
          int xw_floor = (int)floor(x_w_data[ index_x ]);
          int yw_floor = (int)floor(x_w_data[ index_y ]);
          int xw_ceil = (int)ceil(x_w_data[ index_x ]);
          int yw_ceil = (int)ceil(x_w_data[ index_y ]);

          if (outliers_ == WarpParameter_WarpType_NEAREST) {
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
          }

          for (int c=0; c<channels_; c++) {
            int bottom_0_index = ((n * channels_ + c) * height_ +  h) * width_ +  w;
            int offset = (n * channels_ + c) * height_;
            Dtype I0 = bottom_0_data[ (offset + xw_floor) * width_ + yw_floor ]; 
            Dtype I1 = bottom_0_data[ (offset + xw_ceil ) * width_ + yw_floor ]; 
            Dtype I2 = bottom_0_data[ (offset + xw_floor) * width_ + yw_ceil ]; 
            Dtype I3 = bottom_0_data[ (offset + xw_ceil ) * width_ + yw_ceil ];
            if (!(outliers_ == WarpParameter_WarpType_TRUNCATE && 
                  (x_w_data[ index_x ] < 0 ||
                  x_w_data[ index_x ] > height_-1 ||
                  x_w_data[ index_y ] < 0 ||
                  x_w_data[ index_y ] > width_-1))) {
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
              bottom_0_diff[ (offset + xw_floor) * width_ + yw_floor ] += theta_data_[ index_x ]*theta_data_[ index_y ]*top_diff[bottom_0_index];
              bottom_0_diff[ (offset + xw_ceil ) * width_ + yw_floor ] += theta_data[ index_x ] *theta_data_[ index_y ]*top_diff[bottom_0_index];
              bottom_0_diff[ (offset + xw_floor) * width_ + yw_ceil  ] += theta_data_[ index_x ]*theta_data[ index_y ] *top_diff[bottom_0_index];
              bottom_0_diff[ (offset + xw_ceil ) * width_ + yw_ceil  ] += theta_data[ index_x ] *theta_data[ index_y ] *top_diff[bottom_0_index];
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WarpLayer);
#endif

INSTANTIATE_CLASS(WarpLayer);
REGISTER_LAYER_CLASS(Warp);

}  // namespace caffe

