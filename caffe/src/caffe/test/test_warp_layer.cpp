// Copyright 2017 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/warp_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class WarpLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  WarpLayerTest()
      : blob_bottom_a_0_(new Blob<Dtype>(1, 1, 4, 5)),
        blob_bottom_a_1_(new Blob<Dtype>(1, 2, 4, 5)),
        blob_bottom_b_0_(new Blob<Dtype>(1, 1, 3, 4)),
        blob_bottom_b_1_(new Blob<Dtype>(1, 2, 3, 4)),
        blob_bottom_c_0_(new Blob<Dtype>(1, 1, 3, 4)),
        blob_bottom_c_1_(new Blob<Dtype>(1, 2, 3, 4)),
        blob_bottom_d_0_(new Blob<Dtype>(1, 1, 2, 2)),
        blob_bottom_d_1_(new Blob<Dtype>(1, 2, 2, 2)),
        blob_bottom_e_0_(new Blob<Dtype>(1, 1, 2, 3)),
        blob_bottom_e_1_(new Blob<Dtype>(1, 2, 2, 3)),
        blob_bottom_f_0_(new Blob<Dtype>(1, 1, 2, 2)),
        blob_bottom_f_1_(new Blob<Dtype>(1, 2, 2, 2)),
        blob_bottom_g_0_(new Blob<Dtype>(1, 1, 2, 2)),
        blob_bottom_g_1_(new Blob<Dtype>(1, 2, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    blob_bottom_vec_1_.push_back(blob_bottom_a_0_);
    blob_bottom_vec_1_.push_back(blob_bottom_a_1_);
    blob_bottom_vec_2_.push_back(blob_bottom_b_0_);
    blob_bottom_vec_2_.push_back(blob_bottom_b_1_);
    blob_bottom_vec_3_.push_back(blob_bottom_c_0_);
    blob_bottom_vec_3_.push_back(blob_bottom_c_1_);
    blob_bottom_vec_4_.push_back(blob_bottom_d_0_);
    blob_bottom_vec_4_.push_back(blob_bottom_d_1_);
    blob_bottom_vec_5_.push_back(blob_bottom_e_0_);
    blob_bottom_vec_5_.push_back(blob_bottom_e_1_);
    blob_bottom_vec_6_.push_back(blob_bottom_f_0_);
    blob_bottom_vec_6_.push_back(blob_bottom_f_1_);
    blob_bottom_vec_7_.push_back(blob_bottom_g_0_);
    blob_bottom_vec_7_.push_back(blob_bottom_g_1_);
    blob_top_vec_.push_back(blob_top_);
    for (int i = 0; i < blob_bottom_a_0_->count(); i++)
        blob_bottom_a_0_->mutable_cpu_data()[i] = i;
    for (int i = 0; i < blob_bottom_b_0_->count(); i++)
        blob_bottom_b_0_->mutable_cpu_data()[i] = i/100.0;
    for (int i = 0; i < blob_bottom_c_0_->count(); i++)
        blob_bottom_c_0_->mutable_cpu_data()[i] = i/10.0;
    for (int i = 0; i < blob_bottom_d_0_->count(); i++)
        blob_bottom_d_0_->mutable_cpu_data()[i] = (i+5)/10.0;
    for (int i = 0; i < blob_bottom_e_0_->count(); i++)
        blob_bottom_e_0_->mutable_cpu_data()[i] = (i+5)/10.0;
    for (int i = 0; i < blob_bottom_f_0_->count(); i++)
        blob_bottom_f_0_->mutable_cpu_data()[i] = (i+5)/10.0;
    for (int i = 0; i < blob_bottom_g_0_->count(); i++)
        blob_bottom_f_0_->mutable_cpu_data()[i] = 0.0;
    blob_bottom_f_0_->mutable_cpu_data()[0] = 0.5;
    
    blob_bottom_a_1_->mutable_cpu_data()[0] =-1.0135;
    blob_bottom_a_1_->mutable_cpu_data()[1] =-1.0031;
    blob_bottom_a_1_->mutable_cpu_data()[2] = 0.5676;
    blob_bottom_a_1_->mutable_cpu_data()[3] =-0.8131;
    blob_bottom_a_1_->mutable_cpu_data()[4] = 0.1150;
    blob_bottom_a_1_->mutable_cpu_data()[5] = 0.8829;
    blob_bottom_a_1_->mutable_cpu_data()[6] = 0.3059;
    blob_bottom_a_1_->mutable_cpu_data()[7] = 0.7445;
    blob_bottom_a_1_->mutable_cpu_data()[8] = 1.2400;
    blob_bottom_a_1_->mutable_cpu_data()[9] = 1.4884;
    blob_bottom_a_1_->mutable_cpu_data()[10] =-0.5664;
    blob_bottom_a_1_->mutable_cpu_data()[11] =-0.7111;
    blob_bottom_a_1_->mutable_cpu_data()[12] =-0.1484;
    blob_bottom_a_1_->mutable_cpu_data()[13] =-1.0429;
    blob_bottom_a_1_->mutable_cpu_data()[14] =-1.2655;
    blob_bottom_a_1_->mutable_cpu_data()[15] = 0.0856;
    blob_bottom_a_1_->mutable_cpu_data()[16] = 0.4622;
    blob_bottom_a_1_->mutable_cpu_data()[17] =-1.2485;
    blob_bottom_a_1_->mutable_cpu_data()[18] = 0.9775;
    blob_bottom_a_1_->mutable_cpu_data()[19] =-0.1720;

    blob_bottom_a_1_->mutable_cpu_data()[20] =-1.1800;
    blob_bottom_a_1_->mutable_cpu_data()[21] = 0.9519;
    blob_bottom_a_1_->mutable_cpu_data()[22] =-0.7204;
    blob_bottom_a_1_->mutable_cpu_data()[23] =-0.9545;
    blob_bottom_a_1_->mutable_cpu_data()[24] = 1.1079;
    blob_bottom_a_1_->mutable_cpu_data()[25] = 1.3857;
    blob_bottom_a_1_->mutable_cpu_data()[26] = 1.1061;
    blob_bottom_a_1_->mutable_cpu_data()[27] = 0.9002;
    blob_bottom_a_1_->mutable_cpu_data()[28] =-0.7086;
    blob_bottom_a_1_->mutable_cpu_data()[29] = 0.2391;
    blob_bottom_a_1_->mutable_cpu_data()[30] =-1.4861;
    blob_bottom_a_1_->mutable_cpu_data()[31] =-1.2467;
    blob_bottom_a_1_->mutable_cpu_data()[32] =-0.2058;
    blob_bottom_a_1_->mutable_cpu_data()[33] =-1.0634;
    blob_bottom_a_1_->mutable_cpu_data()[34] = 0.1496;
    blob_bottom_a_1_->mutable_cpu_data()[35] = 0.8247;
    blob_bottom_a_1_->mutable_cpu_data()[36] =-0.3007;
    blob_bottom_a_1_->mutable_cpu_data()[37] = 1.2319;
    blob_bottom_a_1_->mutable_cpu_data()[38] =-1.0918;
    blob_bottom_a_1_->mutable_cpu_data()[39] =-1.0651;

    blob_bottom_b_1_->mutable_cpu_data()[0] = 0.6948;
    blob_bottom_b_1_->mutable_cpu_data()[1] = 0.0344;
    blob_bottom_b_1_->mutable_cpu_data()[2] = 0.7655;
    blob_bottom_b_1_->mutable_cpu_data()[3] = 0.4898;
    blob_bottom_b_1_->mutable_cpu_data()[4] = 0.3171;
    blob_bottom_b_1_->mutable_cpu_data()[5] = 0.4387;
    blob_bottom_b_1_->mutable_cpu_data()[6] = 0.7952;
    blob_bottom_b_1_->mutable_cpu_data()[7] = 0.4456;
    blob_bottom_b_1_->mutable_cpu_data()[8] =-0.0498;
    blob_bottom_b_1_->mutable_cpu_data()[9] =-0.6184;
    blob_bottom_b_1_->mutable_cpu_data()[10] =-0.8131;
    blob_bottom_b_1_->mutable_cpu_data()[11] =-0.3537;

    blob_bottom_b_1_->mutable_cpu_data()[12] = 0.9572;
    blob_bottom_b_1_->mutable_cpu_data()[13] = 0.1419;
    blob_bottom_b_1_->mutable_cpu_data()[14] = 0.7922;
    blob_bottom_b_1_->mutable_cpu_data()[15] =-0.9643;
    blob_bottom_b_1_->mutable_cpu_data()[16] = 0.4854;
    blob_bottom_b_1_->mutable_cpu_data()[17] = 0.4218;
    blob_bottom_b_1_->mutable_cpu_data()[18] = 0.9595;
    blob_bottom_b_1_->mutable_cpu_data()[19] =-0.1509;
    blob_bottom_b_1_->mutable_cpu_data()[20] = 0.8003;
    blob_bottom_b_1_->mutable_cpu_data()[21] = 0.9157;
    blob_bottom_b_1_->mutable_cpu_data()[22] = 0.6557;
    blob_bottom_b_1_->mutable_cpu_data()[23] =-0.0660;

    blob_bottom_c_1_->mutable_cpu_data()[0] = 0.6948;
    blob_bottom_c_1_->mutable_cpu_data()[1] = 0.0344;
    blob_bottom_c_1_->mutable_cpu_data()[2] = 0.7655;
    blob_bottom_c_1_->mutable_cpu_data()[3] = 0.4898;
    blob_bottom_c_1_->mutable_cpu_data()[4] = 0.3171;
    blob_bottom_c_1_->mutable_cpu_data()[5] = 0.4387;
    blob_bottom_c_1_->mutable_cpu_data()[6] = 0.7952;
    blob_bottom_c_1_->mutable_cpu_data()[7] = 0.4456;
    blob_bottom_c_1_->mutable_cpu_data()[8] =-0.0498;
    blob_bottom_c_1_->mutable_cpu_data()[9] =-0.6184;
    blob_bottom_c_1_->mutable_cpu_data()[10] =-0.8131;
    blob_bottom_c_1_->mutable_cpu_data()[11] =-0.3537;

    blob_bottom_c_1_->mutable_cpu_data()[12] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[13] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[14] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[15] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[16] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[17] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[18] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[19] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[20] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[21] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[22] = 0;
    blob_bottom_c_1_->mutable_cpu_data()[23] = 0;

    blob_bottom_d_1_->mutable_cpu_data()[0] = 0.2;
    blob_bottom_d_1_->mutable_cpu_data()[1] =-0.2;
    blob_bottom_d_1_->mutable_cpu_data()[2] = 0.2;
    blob_bottom_d_1_->mutable_cpu_data()[3] =-0.2;
    blob_bottom_d_1_->mutable_cpu_data()[4] = 0.3;
    blob_bottom_d_1_->mutable_cpu_data()[5] = 0.3;
    blob_bottom_d_1_->mutable_cpu_data()[6] =-0.3;
    blob_bottom_d_1_->mutable_cpu_data()[7] =-0.3;

    blob_bottom_e_1_->mutable_cpu_data()[0] = 1.2;
    blob_bottom_e_1_->mutable_cpu_data()[1] = 0.2;
    blob_bottom_e_1_->mutable_cpu_data()[2] =-0.2;
    blob_bottom_e_1_->mutable_cpu_data()[3] = 1.2;
    blob_bottom_e_1_->mutable_cpu_data()[4] = 0.2;
    blob_bottom_e_1_->mutable_cpu_data()[5] =-0.2;
    blob_bottom_e_1_->mutable_cpu_data()[6] = 0.3;
    blob_bottom_e_1_->mutable_cpu_data()[7] = 0.3;
    blob_bottom_e_1_->mutable_cpu_data()[8] = 0.3;
    blob_bottom_e_1_->mutable_cpu_data()[9] =-0.3;
    blob_bottom_e_1_->mutable_cpu_data()[10] =-0.3;
    blob_bottom_e_1_->mutable_cpu_data()[11] =-0.3;

    blob_bottom_f_1_->mutable_cpu_data()[0] =-0.2;
    blob_bottom_f_1_->mutable_cpu_data()[1] =-0.2;
    blob_bottom_f_1_->mutable_cpu_data()[2] =-0.2;
    blob_bottom_f_1_->mutable_cpu_data()[3] =-0.2;
    blob_bottom_f_1_->mutable_cpu_data()[4] =-0.3;
    blob_bottom_f_1_->mutable_cpu_data()[5] = 0.3;
    blob_bottom_f_1_->mutable_cpu_data()[6] =-0.3;
    blob_bottom_f_1_->mutable_cpu_data()[7] =-0.3;

    for (int i = 0; i < blob_bottom_g_1_->count(); i++)
        blob_bottom_g_1_->mutable_cpu_data()[i] = blob_bottom_d_1_->mutable_cpu_data()[i];
  }
  virtual ~WarpLayerTest() {
    delete blob_bottom_a_0_;
    delete blob_bottom_a_1_;
    delete blob_bottom_b_0_;
    delete blob_bottom_b_1_;
    delete blob_bottom_c_0_;
    delete blob_bottom_c_1_;
    delete blob_bottom_d_0_;
    delete blob_bottom_d_1_;
    delete blob_bottom_e_0_;
    delete blob_bottom_e_1_;
    delete blob_bottom_f_0_;
    delete blob_bottom_f_1_;
    delete blob_bottom_g_0_;
    delete blob_bottom_g_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_a_0_;
  Blob<Dtype>* const blob_bottom_a_1_;
  Blob<Dtype>* const blob_bottom_b_0_;
  Blob<Dtype>* const blob_bottom_b_1_;
  Blob<Dtype>* const blob_bottom_c_0_;
  Blob<Dtype>* const blob_bottom_c_1_;
  Blob<Dtype>* const blob_bottom_d_0_;
  Blob<Dtype>* const blob_bottom_d_1_;
  Blob<Dtype>* const blob_bottom_e_0_;
  Blob<Dtype>* const blob_bottom_e_1_;
  Blob<Dtype>* const blob_bottom_f_0_;
  Blob<Dtype>* const blob_bottom_f_1_;
  Blob<Dtype>* const blob_bottom_g_0_;
  Blob<Dtype>* const blob_bottom_g_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_bottom_vec_3_;
  vector<Blob<Dtype>*> blob_bottom_vec_4_;
  vector<Blob<Dtype>*> blob_bottom_vec_5_;
  vector<Blob<Dtype>*> blob_bottom_vec_6_;
  vector<Blob<Dtype>*> blob_bottom_vec_7_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WarpLayerTest, TestDtypesAndDevices);

TYPED_TEST(WarpLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);

  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(WarpLayerTest, TestNearestForward1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  EXPECT_NEAR(data[0],      0., 1e-3);
  EXPECT_NEAR(data[1],  4.7595, 1e-3);
  EXPECT_NEAR(data[2],  2.5676, 1e-3);
  EXPECT_NEAR(data[3],  2.1869, 1e-3);
  EXPECT_NEAR(data[4],  9.5394, 1e-3);
  EXPECT_NEAR(data[5], 12.8113, 1e-3);
  EXPECT_NEAR(data[6], 11.8364, 1e-3);
  EXPECT_NEAR(data[7], 12.2455, 1e-3);
  EXPECT_NEAR(data[8],  5.4570, 1e-3);
  EXPECT_NEAR(data[9], 10.1956, 1e-3);
  EXPECT_NEAR(data[10],  2.5695, 1e-3);
  EXPECT_NEAR(data[11],  4.0555, 1e-3);
  EXPECT_NEAR(data[12], 10.8228, 1e-3);
  EXPECT_NEAR(data[13],  6.6402, 1e-3);
  EXPECT_NEAR(data[14], 13.4824, 1e-3);
  EXPECT_NEAR(data[15], 15.0856, 1e-3);
  EXPECT_NEAR(data[16], 14.9590, 1e-3);
  EXPECT_NEAR(data[17], 15.7515, 1e-3);
  EXPECT_NEAR(data[18], 13.5185, 1e-3);
  EXPECT_NEAR(data[19], 13.5024, 1e-3);
}

TYPED_TEST(WarpLayerTest, TestNearestForward2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_2_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.0452, 1e-4);
  EXPECT_NEAR(data[1],0.0160, 1e-4);
  EXPECT_NEAR(data[2],0.0593, 1e-4);
  EXPECT_NEAR(data[3],0.0300, 1e-4);
  EXPECT_NEAR(data[4],0.0626, 1e-4);
  EXPECT_NEAR(data[5],0.0713, 1e-4);
  EXPECT_NEAR(data[6],0.1063, 1e-4);
  EXPECT_NEAR(data[7],0.0640, 1e-4);
  EXPECT_NEAR(data[8],0.0800, 1e-4);
  EXPECT_NEAR(data[9],0.0838, 1e-4);
  EXPECT_NEAR(data[10],0.0919, 1e-4);
  EXPECT_NEAR(data[11],0.1038, 1e-4);
}

TYPED_TEST(WarpLayerTest, TestNearestForward3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_3_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.0695, 1e-4);
  EXPECT_NEAR(data[1],0.1034, 1e-4);
  EXPECT_NEAR(data[2],0.2766, 1e-4);
  EXPECT_NEAR(data[3],0.3000, 1e-4);
  EXPECT_NEAR(data[4],0.4317, 1e-4);
  EXPECT_NEAR(data[5],0.5439, 1e-4);
  EXPECT_NEAR(data[6],0.6795, 1e-4);
  EXPECT_NEAR(data[7],0.7000, 1e-4);
  EXPECT_NEAR(data[8],0.8000, 1e-4);
  EXPECT_NEAR(data[9],0.8382, 1e-4);
  EXPECT_NEAR(data[10],0.9187, 1e-4);
  EXPECT_NEAR(data[11],1.0646, 1e-4);
}

TYPED_TEST(WarpLayerTest, TestNearestForward4) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_4_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_4_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.58, 1e-4);
  EXPECT_NEAR(data[1],0.64, 1e-4);
  EXPECT_NEAR(data[2],0.66, 1e-4);
  EXPECT_NEAR(data[3],0.72, 1e-4);
}

TYPED_TEST(WarpLayerTest, TestNearestForward5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_5_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_5_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.71, 1e-4);
  EXPECT_NEAR(data[1],0.71, 1e-4);
  EXPECT_NEAR(data[2],0.77, 1e-4);
  EXPECT_NEAR(data[3],0.83, 1e-4);
  EXPECT_NEAR(data[4],0.83, 1e-4);
  EXPECT_NEAR(data[5],0.89, 1e-4);
}

/*TYPED_TEST(WarpLayerTest, TestNearestForward6) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_6_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_6_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.5 , 1e-4);
  EXPECT_NEAR(data[1],0.64, 1e-4);
  EXPECT_NEAR(data[2],0.64, 1e-4);
  EXPECT_NEAR(data[3],0.72, 1e-4);
}*/

TYPED_TEST(WarpLayerTest, TestTruncateForward1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_1_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  EXPECT_NEAR(data[0],      0., 1e-3);
  EXPECT_NEAR(data[1],      0., 1e-3);
  EXPECT_NEAR(data[2],      0., 1e-3);
  EXPECT_NEAR(data[3],      0., 1e-3);
  EXPECT_NEAR(data[4],      0., 1e-3);
  EXPECT_NEAR(data[5], 12.8113, 1e-3);
  EXPECT_NEAR(data[6], 11.8364, 1e-3);
  EXPECT_NEAR(data[7], 12.2455, 1e-3);
  EXPECT_NEAR(data[8],      0., 1e-3);
  EXPECT_NEAR(data[9],      0., 1e-3);
  EXPECT_NEAR(data[10],      0., 1e-3);
  EXPECT_NEAR(data[11],  4.0555, 1e-3);
  EXPECT_NEAR(data[12], 10.8228, 1e-3);
  EXPECT_NEAR(data[13],  6.6402, 1e-3);
  EXPECT_NEAR(data[14], 13.4824, 1e-3);
  EXPECT_NEAR(data[15],      0., 1e-3);
  EXPECT_NEAR(data[16], 14.9590, 1e-3);
  EXPECT_NEAR(data[17],      0., 1e-3);
  EXPECT_NEAR(data[18], 13.5185, 1e-3);
  EXPECT_NEAR(data[19], 13.5024, 1e-3);
}

TYPED_TEST(WarpLayerTest, TestTruncateForward2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_2_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_2_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.0452, 1e-4);
  EXPECT_NEAR(data[1],0.0160, 1e-4);
  EXPECT_NEAR(data[2],0.0593, 1e-4);
  EXPECT_NEAR(data[3],    0., 1e-4);
  EXPECT_NEAR(data[4],0.0626, 1e-4);
  EXPECT_NEAR(data[5],0.0713, 1e-4);
  EXPECT_NEAR(data[6],0.1063, 1e-4);
  EXPECT_NEAR(data[7],    0., 1e-4);
  EXPECT_NEAR(data[8],    0., 1e-4);
  EXPECT_NEAR(data[9],    0., 1e-4);
  EXPECT_NEAR(data[10],    0., 1e-4);
  EXPECT_NEAR(data[11],0.1038, 1e-4);
}

TYPED_TEST(WarpLayerTest, TestTruncateForward3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_3_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_3_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.0695, 1e-4);
  EXPECT_NEAR(data[1],0.1034, 1e-4);
  EXPECT_NEAR(data[2],0.2766, 1e-4);
  EXPECT_NEAR(data[3],    0., 1e-4);
  EXPECT_NEAR(data[4],0.4317, 1e-4);
  EXPECT_NEAR(data[5],0.5439, 1e-4);
  EXPECT_NEAR(data[6],0.6795, 1e-4);
  EXPECT_NEAR(data[7],    0., 1e-4);
  EXPECT_NEAR(data[8],    0., 1e-4);
  EXPECT_NEAR(data[9],0.8382, 1e-4);
  EXPECT_NEAR(data[10],0.9187, 1e-4);
  EXPECT_NEAR(data[11],1.0646, 1e-4);
}

TYPED_TEST(WarpLayerTest, TestTruncateForward4) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_4_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_4_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  //for (int i=0;i<4;i++) std::cout << data[i] << " "; std::cout << "\n";
  // for every top blob element ...
  EXPECT_NEAR(data[0],0.58, 1e-4);
  EXPECT_NEAR(data[1],0.64, 1e-4);
  EXPECT_NEAR(data[2],0.66, 1e-4);
  EXPECT_NEAR(data[3],0.72, 1e-4);
}

TYPED_TEST(WarpLayerTest, TestTruncateForward5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_5_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_5_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0.71, 1e-4);
  EXPECT_NEAR(data[1],0.71, 1e-4);
  EXPECT_NEAR(data[2],0.77, 1e-4);
  EXPECT_NEAR(data[3],0.83, 1e-4);
  EXPECT_NEAR(data[4],0.83, 1e-4);
  EXPECT_NEAR(data[5],0.89, 1e-4);
}

/*TYPED_TEST(WarpLayerTest, TestTruncateForward6) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  shared_ptr<WarpLayer<Dtype> > layer(
      new WarpLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_6_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_6_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  // for every top blob element ...
  EXPECT_NEAR(data[0],0. , 1e-4);
  EXPECT_NEAR(data[1],0.64, 1e-4);
  EXPECT_NEAR(data[2],0.  , 1e-4);
  EXPECT_NEAR(data[3],0.72, 1e-4);
}*/

TYPED_TEST(WarpLayerTest, TestNearestFlowGradient1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_1_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestNearestFlowGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 1);
}

/*TYPED_TEST(WarpLayerTest, TestNearestFlowGradient3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_3_,
      this->blob_top_vec_, 1);
}*/

TYPED_TEST(WarpLayerTest, TestNearestFlowGradient4) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_4_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestNearestFlowGradient5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_5_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestNearestFlowGradient6) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_6_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestTruncateFlowGradient1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-5, 1e-1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_1_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestTruncateFlowGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 1);
}

/*TYPED_TEST(WarpLayerTest, TestTruncateFlowGradient3) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_3_,
      this->blob_top_vec_, 1);
}*/

TYPED_TEST(WarpLayerTest, TestTruncateFlowGradient4) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_4_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestTruncateFlowGradient5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_5_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestTruncateFlowGradient6) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_6_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(WarpLayerTest, TestNearestImgGradient1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_1_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestNearestImgGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestNearestImgGradient4) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_4_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestNearestImgGradient5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_5_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestNearestImgGradient6) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_6_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestNearestImgGradient7) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_NEAREST);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_7_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestTruncateImgGradient1) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-1, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_1_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestTruncateImgGradient2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestTruncateImgGradient4) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_4_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestTruncateImgGradient5) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_5_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestTruncateImgGradient6) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_6_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(WarpLayerTest, TestTruncateImgGradient7) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  WarpParameter* warp_param = layer_param.mutable_warp_param();
  warp_param->set_outliers(WarpParameter_WarpType_TRUNCATE);
  WarpLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_7_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
