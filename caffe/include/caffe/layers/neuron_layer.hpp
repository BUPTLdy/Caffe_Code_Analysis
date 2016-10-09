#ifndef CAFFE_NEURON_LAYER_HPP_
#define CAFFE_NEURON_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief An interface for layers that take one blob as input (@f$ x @f$)
 *        and produce one equally-sized blob as output (@f$ y @f$), where
 *        each element of the output depends only on the corresponding input
 *        element.
 *输入了data后，就要计算了，比如常见的sigmoid、
 *tanh等等。这些都计算操作被抽象成了neuron_layers.hpp里面的类
 *NeuronLayer，这个层只负责具体的计算，因此明确定义了输入
 *ExactNumBottomBlobs()和ExactNumTopBlobs()都是常量1,即输入一个
 *blob，输出一个blob。 其派生类主要是元素级别的运算（比如
 *Dropout运算，激活函数ReLu， Sigmoid等），运算均为同址计算（ inplace
 *computation，返回值覆盖原值而占用新的内存）。
 *
 */
template <typename Dtype>

//neuron_layer实现里大量激活函数，主要是元素级别的操作
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
};

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYER_HPP_
