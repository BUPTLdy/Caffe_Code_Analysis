#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"  //单例化caffe类，并且封装了boost和cuda随机数生成的函数，提供了统一接口
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

/*
主要是分配内存和释放内存。class yncedMemory定义了内存分配管理和CPU与GPU之间同步的函数。
 Blob会使用SyncedMem自动决定什么时候去copy data以提高运行效率，通常情况是仅当gnu或cpu修改后有copy操作。
 */

const int kMaxBlobAxes = 32; //在头文件中为它添加 extern 声明,以使其能被多个文件共享

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>   //模板类，虚拟类型Dtype
class Blob {
 public:
  Blob() //构造函数：初始化列表 {空函数体}
       : data_(), diff_(), count_(0), capacity_(0) {}


  //当构造函数被声明 explicit 时,编译器将不使用它作为转换操作符。
  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);   //可以通过设置数据维度（N,C,H,W）初始化
  //const 传递过来的参数在函数内不可以改变(无意义，因为本身就是形参)


  //const引用参数在函数内为常量不可变
  explicit Blob(const vector<int>& shape); //也可以通过传入vector<int>直接传入维数

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  void Reshape(const int num, const int channels, const int height,
      const int width);


  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   *
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */

  void Reshape(const vector<int>& shape);
  void Reshape(const BlobShape& shape);
  void ReshapeLike(const Blob& other);


  //内联函数 通过内联函数，编译器不需要跳转到内存其他地址去执行函数调用，也不需要保留函数调用时的现场数据。
  // const 成员函数，任何不会修改数据成员的函数都应该声明为const 类型。

  // 输出blob的形状
  inline string shape_string() const { //
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }

  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */
  inline int shape(int index) const {  //根据索引返回维数，对于维数(N,C,H,W),shape(0)返回N,shape(-1)返回W。
    return shape_[CanonicalAxisIndex(index)];
  }

  inline int num_axes() const { return shape_.size(); }    //返回Blob维度数，对于维数(N,C,H,W)，返回4

  inline int count() const { return count_; }               //返回Blob维度数，对于维数(N,C,H,W)，返回N×C×H×W

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
  //对于维数(N,C,H,W)，count(0, 3)返回N×C×H
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }


  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */
  //对于维数(N,C,H,W)，count(1)返回C×H×W
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }


  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }



  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }



  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {    //计算物理偏移量，(n,c,h,w)的偏移量为((n∗C+c)∗H+h)∗W+w
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }


  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff: if false, copy the data; if true, copy the diff
   * @param reshape: if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,   //从source拷贝数据， copy_diff来作为标志区分是拷贝data还是diff。
      bool reshape = false);

  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }\


  /*
  	// 假定数据在 CPU 上进行初始化，我们有一个 blob
	const Dtype* foo;
	Dtype* bar;
	foo = blob.gpu_data(); // 数据从 CPU 复制到 GPU
	foo = blob.cpu_data(); // 没有数据复制，两者都有最新的内容
	bar = blob.mutable_gpu_data(); // 没有数据复制
	// ... 一些操作 ...
	bar = blob.mutable_gpu_data(); // 仍在 GPU，没有数据复制
	foo = blob.cpu_data(); // 由于 GPU 修改了数值，数据从 GPU 复制到 CPU
	foo = blob.gpu_data(); //没有数据复制，两者都有最新的内容
	bar = blob.mutable_cpu_data(); // 依旧没有数据复制
	bar = blob.mutable_gpu_data(); //数据从 CPU 复制到 GPU
	bar = blob.mutable_cpu_data(); //数据从 GPU 复制到 CPU

   */

  const Dtype* cpu_data() const;  //数据访问，const方式只读，不允许改写数据
  void set_cpu_data(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();      //mutable方式可改写数据（对diff_的访问也是类似的）
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
  void Update();

  void FromProto(const BlobProto& proto, bool reshape = true); //从proto读数据进来，其实就是反序列化
  void ToProto(BlobProto* proto, bool write_diff = false) const; //blob数据保存到proto中

  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other); //Blob& other 赋值给data_

  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other); //Blob& other 赋值给diff_

  bool ShapeEquals(const BlobProto& other);

 protected:
  shared_ptr<SyncedMemory> data_; //存储前向传递数据
  shared_ptr<SyncedMemory> diff_; //存储反向传递梯度
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;  //参数维度
  int count_; //Blob存储的元素个数（shape_所有元素乘积）
  int capacity_;//当前Blob的元素个数（控制动态分配）

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
