// BSD 3-Clause License
//
// Copyright (c) 2022-2025, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
template <typename T>
__global__ void pool_index(Cize IY, Cize IX, Cize OY, Cize OX, Cize KY, Cize KX, Cize SY, Cize SX,  //
                           PTR(T, input), PTR(I32, output))
{
    input += blockIdx.z * IY * IX, output += blockIdx.z * OY * OX;
    Cize y = Index2Y, x = Index2X;
    if (y >= OY || x >= OX) return;

    auto index = 0;
    T value = Limits<T>::min();
    for (Cize i = 0; i < KY; ++i)
        for (Cize j = 0; j < KX; ++j)
        {
            I32 idx = (y * SY + i) * IX + (x * SX + j);
            if (value < input[idx]) value = input[idx], index = idx;
        }
    output[y * OX + x] = index;
}

template <typename A>
__global__ void rate_pool(Cize T, Cize C, Cize I, Cize O, PTR(A, input), PTR(I32, index), PTR(A, output))
{
    input += (blockIdx.z * C + blockIdx.y) * I;
    output += (blockIdx.z * C + blockIdx.y) * O;
    index += ((blockIdx.z / T) * C + blockIdx.y) * O;
    Cize idx = Index1D(A), end = min(O, idx + Block1D(A));
    for (Cize i = idx; i < end; i += Thread1D) output[i] = input[index[i]];
}

template <typename T, typename R>
void rate_pool(Vec5<T> input, Vec4<R> rates, Vec5<T> output, Len2 kernel, Len2 stride, Len4 pad)
{
    SpykerAssert(pad.t == 0 && pad.z == 0 && pad.y == 0 && pad.x == 0, "CPU::Pool",
                 "Padding is not supported in rate pooling.");

    auto index = init<I32>(output.u, output.z, output.y, output.x);
    pool_index<<<Config2(index.t * index.z, index.y, index.x)>>>  //
        (rates.y, rates.x, index.y, index.x, kernel.y, kernel.x, stride.y, stride.x, rates.data, index.data);
    rate_pool<<<Config1D(T, output.u * output.t, output.z, output.y * output.x)>>>  //
        (output.t, output.z, input.y * input.x, output.y * output.x, input.data, index.data, output.data);
    deinit(index);
}

#ifndef SPYKER_USE_CUDNN
template <typename T>
__global__ void rank_pool_(Cize IY, Cize IX, Cize OY, Cize OX, Cize KY, Cize KX, Cize SY, Cize SX,  //
                           PTR(T, input), PTR(T, output))
{
    input += blockIdx.z * IY * IX, output += blockIdx.z * OY * OX;
    Cize y = Index2Y, x = Index2X;
    if (y >= OY || x >= OX) return;

    T value = Limits<T>::min();
    for (Cize i = 0; i < KY; ++i)
        for (Cize j = 0; j < KX; ++j)  //
            value = cmax(value, input[(y * SY + i) * IX + (x * SX + j)]);
    output[y * OX + x] = value;
}

template <typename T>
void rank_pool_(Vec4<T> input, Vec4<T> output, Len2 kernel, Len2 stride, Len4)
{
    rank_pool_<<<Config2(output.t * output.z, output.y, output.x)>>>  //
        (input.y, input.x, output.y, output.x, kernel.y, kernel.x, stride.y, stride.x, input.data, output.data);
}

#else

struct Pool
{
    cudnnTensorDescriptor_t input;
    cudnnTensorDescriptor_t output;
    cudnnPoolingDescriptor_t pool;

    Len2 _kernel;
    Len2 _stride;
    Len2 _pad;

    Pool(Len2 _kernel, Len2 _stride, Len2 _pad) : _kernel(_kernel), _stride(_stride), _pad(_pad)
    {
        if (!cudnn_static) cudnn_static = std::unique_ptr<cudnn>(new cudnn);
        CudnnCheck(cudnnCreateTensorDescriptor(&input));
        CudnnCheck(cudnnCreateTensorDescriptor(&output));
        CudnnCheck(cudnnCreatePoolingDescriptor(&pool));
        CudnnCheck(cudnnSetPooling2dDescriptor(                              //
            pool, CUDNN_POOLING_MAX_DETERMINISTIC, CUDNN_NOT_PROPAGATE_NAN,  //
            _kernel.y, _kernel.x, _pad.y, _pad.x, _stride.y, _stride.x));
    }
    ~Pool()
    {
        cudnnDestroyTensorDescriptor(input);
        cudnnDestroyTensorDescriptor(output);
        cudnnDestroyPoolingDescriptor(pool);
    }
    cudnnDataType_t cvt(Type type)
    {
        if (type == Type::I8) return CUDNN_DATA_INT8;
        if (type == Type::U8) return CUDNN_DATA_INT8;
        if (type == Type::F16) return CUDNN_DATA_HALF;
        if (type == Type::F32) return CUDNN_DATA_FLOAT;
        if (type == Type::F64) return CUDNN_DATA_DOUBLE;
        SpykerAssert(false, "CUDA::Conv", "Data type is not supported in cuDNN.");
    }
    template <typename T>
    void operator()(Vec4<T> input_, Vec4<T> output_)
    {
        F32 alpha = 1, beta = 0;
        CudnnCheck(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, cvt(TypeName<T>()),  //
                                              input_.t, input_.z, input_.y, input_.x));
        CudnnCheck(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, cvt(TypeName<T>()),  //
                                              output_.t, output_.z, output_.y, output_.x));
        CudnnCheck(cudnnPoolingForward(cudnn_static->handle, pool, &alpha, input, input_.data,  //
                                       &beta, output, output_.data));
    }
    void operator()(Vec4<F64> input_, Vec4<F64> output_)
    {
        F64 alpha = 1, beta = 0;
        CudnnCheck(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,  //
                                              input_.t, input_.z, input_.y, input_.x));
        CudnnCheck(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,  //
                                              output_.t, output_.z, output_.y, output_.x));
        CudnnCheck(cudnnPoolingForward(cudnn_static->handle, pool, &alpha, input, input_.data,  //
                                       &beta, output, output_.data));
    }
    bool comp(Len2 kernel_, Len2 stride_, Len2 pad_)
    {
        return _kernel == kernel_ && _stride == stride_ && _pad == pad_;
    }
};

std::vector<std::shared_ptr<Pool>> pool_handle;

Pool &pool_find(Len2 kernel, Len2 stride, Len4 pad)
{
    auto pad_ = (pad.t != pad.y || pad.z != pad.x) ? Len2(0, 0) : Len2(pad.t, pad.z);
    for (auto pool : pool_handle)
        if (pool->comp(kernel, stride, pad_)) return *pool.get();
    pool_handle.push_back(std::shared_ptr<Pool>(new Pool(kernel, stride, pad_)));
    return *pool_handle.back().get();
}

template <typename T>
void rank_pool_(Vec4<T> input, Vec4<T> output, Len2 kernel, Len2 stride, Len4 pad)
{
    pool_find(kernel, stride, pad)(input, output);
}
#endif

template <typename T>
void pad(Vec3<T> input, Vec3<T> output, Len4 pad, T value)
{
    fill(output, value);
    Size stride = sizeof(T) / sizeof(unsigned char);
    cudaMemcpy3DParms param = {0};
    param.kind = cudaMemcpyDeviceToDevice;
    param.extent = make_cudaExtent(input.x * stride, input.y, input.z);
    param.srcPtr = make_cudaPitchedPtr(input.data, input.x * stride, input.x, input.y);
    param.dstPtr = make_cudaPitchedPtr(output.data, output.x * stride, output.x, output.y);
    param.dstPos = make_cudaPos(pad.z * stride, pad.t, 0);
    cudaMemcpy3D(&param);
}

template <typename T>
void rank_pool(Vec4<T> input_, Vec4<T> output, Len2 kernel, Len2 stride, Len4 pad)
{
#ifdef SPYKER_USE_CUDNN
    bool symm = (pad.t != pad.y || pad.z != pad.x);
#else
    bool symm = true;
#endif

    auto input = input_;
    if (symm && (pad.t != 0 || pad.z != 0 || pad.y != 0 || pad.x != 0))
    {
        input = init<T>(input_.t, input_.z, input_.y + pad.t + pad.y, input_.x + pad.z + pad.x);
        CUDA::pad(Vec3<T>(input_.data, input_.t * input_.z, input_.y, input_.x),
                  Vec3<T>(input.data, input.t * input.z, input.y, input.x), pad, T(0));
    }
    rank_pool_(input, output, kernel, stride, pad);
    if (input.data != input_.data) deinit(input);
}
}  // namespace CUDA

void cuda_pad(Dyn3 input, Dyn3 output, Len4 pad, Scalar value)
{
    IfType(T, input.type, CUDA::pad<T>(input, output, pad, value));
}
void cuda_rank_pool(Dyn4 input, Dyn4 output, Len2 kernel, Len2 stride, Len4 pad)
{
    if (input.type == Type::I8)
        CUDA::rank_pool<I8>(input, output, kernel, stride, pad);
    else if (input.type == Type::U8)
        CUDA::rank_pool<U8>(input, output, kernel, stride, pad);
    else if (input.type == Type::F16)
        CUDA::rank_pool<C16>(input, output, kernel, stride, pad);
    else if (input.type == Type::F32)
        CUDA::rank_pool<F32>(input, output, kernel, stride, pad);
    else if (input.type == Type::F64)
        CUDA::rank_pool<F64>(input, output, kernel, stride, pad);
    else
        SpykerAssert(false, "CUDA::Pool", "Given data type is not supported.");
}
void cuda_rate_pool(Dyn5 input, Dyn4 rates, Dyn5 output, Len2 kernel, Len2 stride, Len4 pad)
{
    IfType(T, input.type, IfType(R, rates.type, CUDA::rate_pool<T Comma R>(input, rates, output, kernel, stride, pad)));
}
}  // namespace Core
}  // namespace Spyker
