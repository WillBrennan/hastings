#pragma once

#include "hastings/libinfer/tensor.h"

namespace libinfer {
namespace detail {
template <class T, Ordering order>
struct CudaTensor {
    CudaTensor(Tensor& tensor)
        : data(tensor.data<T>()), n(tensor.shape().n), c(tensor.shape().c), h(tensor.shape().h), w(tensor.shape().w) {}
    CudaTensor(const Tensor& tensor)
        : data(tensor.data<T>()), n(tensor.shape().n), c(tensor.shape().c), h(tensor.shape().h), w(tensor.shape().w) {}

    int n, c, h, w = 0;
    T* data = nullptr;

    __device__ int index(const int i, const int j, const int y, const int x) const __restrict__ {
        if (order == Ordering::NCHW) {
            return ((i * c + j) * h + y) * w + x;
        } else if (order == Ordering::NHWC) {
            return ((i * h + y) * w + x) * c + j;
        }
        return 0;
    }

    __device__ T& operator()(const int n, const int c, const int y, const int x) __restrict__ { return data[index(n, c, y, x)]; }
    __device__ const T& operator()(const int n, const int c, const int y, const int x) const __restrict__ {
        return data[index(n, c, y, x)];
    }
};

}  // namespace detail
}  // namespace libinfer