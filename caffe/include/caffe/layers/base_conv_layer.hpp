#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  //显示构造函数
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  //设置基本一些基本的参数
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  //计算输出数据的维度
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  // 矩阵乘法
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }

  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  // 滤波器形状
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  // 步长形状
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  /// padding形状
  Blob<int> pad_;
  /// @brief The spatial dimensions of the dilation.

  ///Dilated convolution 的主要贡献就是，如何在去掉池化下采样操作的
  ///同时，而不降低网络的感受野；而 dilated convolution 中的卷积核则
  ///是将卷积核对图像隔一定的距离像素进行卷积运算
  ///http://weibo.com/ttarticle/p/show?id=2309351000014015306501684301
  Blob<int> dilation_;
  /// @brief The spatial dimensions of the convolution input.
  /// 需要经行卷积运算的输入图像维数
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  ///
  vector<int> col_buffer_shape_;

  /// @brief The spatial dimensions of the output.
  /// 经行卷积运算后的输出形状，在派生类中初始化
  /// 默认为H_out,W_out
  vector<int> output_shape_;

  const vector<int>* bottom_shape_;

  // 是计算二维卷积还是三维卷积，对于(N, C, H, W)的输入结果为2
  // 对于(N, C, D, H, W)的输入，结果为3
  int num_spatial_axes_;

  //卷积核维数定义(C_out,C_in,H_k,W_k)
  //输入数据维数定义(N,C_in,H，W)
  //输出数据维数定义(N,C_out,H_out,W_out)


  //默认为C_in*H*W
  int bottom_dim_;
  //默认为C_out*H_out*W_out
  int top_dim_;

  //通道维数，默认为1
  int channel_axis_;
  // batchsize
  int num_;
  int channels_;
  //没什么用，默认为1
  int group_;

  // H_out*W_out
  int out_spatial_dim_;

  // 默认为C_out*C_in*H_k*H_w
  int weight_offset_;

  //C_out
  int num_output_;

  // 是否使用偏置
  bool bias_term_;
  // 是否1*1卷积核
  bool is_1x1_;
  //是否强制使用n维通用卷积
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  // 封装im2col/col2im，然后就不必输入长长的参数
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
  // 如果不是计算n维通用卷积
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    }

    else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    }
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_cpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
    }
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
#endif

  int num_kernels_im2col_;
  int num_kernels_col2im_;



  // 卷积的输出通道数，在参数配置文件中设置
  //C_out
  int conv_out_channels_;

  //C_in
  int conv_in_channels_;

  // 默认为输出H_out*W_out
  int conv_out_spatial_dim_;

  // C_in*H_k*W_k
  int kernel_dim_;

  int col_offset_;
  int output_offset_;
  //im2col使用的存储空间
  Blob<Dtype> col_buffer_;
  //将偏置扩展成矩阵
  Blob<Dtype> bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
