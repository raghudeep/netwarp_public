// Copyright 2017 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#ifndef CAFFE_WARP_LAYER_HPP_
#define CAFFE_WARP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Warp input blob according to given optical flow.
 */
template <typename Dtype>
class WarpLayer : public Layer<Dtype> {
 public:
  explicit WarpLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Warp"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  WarpParameter_WarpType outliers_;
  Blob<Dtype> theta;
  Blob<Dtype> theta_;
  Blob<Dtype> x_w;

  int num_;
  int channels_;
  int height_;
  int width_;
};

}  // namespace caffe

#endif  // CAFFE_WARP_LAYER_HPP_
