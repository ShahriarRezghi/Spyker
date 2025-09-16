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
bool LightConv = false;
bool HeuristicConv = true;
bool ForceCore = false;

#ifdef SPYKER_USE_CUDNN
struct Workspace
{
    Size size = 0;
    void *ptr = nullptr;

    ~Workspace()
    {
        if (ptr != nullptr) cuda_dealloc(ptr);
    }
} space;

std::map<cudnnConvolutionFwdAlgo_t, std::string> algo_map = {
    {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, "IMPLICIT_GEMM"},
    {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, "IMPLICIT_PRECOMP_GEMM"},
    {CUDNN_CONVOLUTION_FWD_ALGO_GEMM, "GEMM"},
    {CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, "DIRECT"},
    {CUDNN_CONVOLUTION_FWD_ALGO_FFT, "ALGO_FFT"},
    {CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, "FFT_TILING"},
    {CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, "WINOGRAD"},
    {CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, "WINOGRAD_NONFUSED"}};

std::map<cudnnMathType_t, std::string> math_map = {
    {CUDNN_DEFAULT_MATH, "DEFAULT_MATH"},
    {CUDNN_TENSOR_OP_MATH, "TENSOR_OP_MATH"},
    {CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, "TENSOR_OP_MATH_ALLOW_CONVERSION"},
    {CUDNN_FMA_MATH, "FMA_MATH"}};

std::map<Type, std::string> type_map = {  //
    {Type::U8, "U8"},   {Type::U16, "U16"}, {Type::U32, "U32"}, {Type::U64, "U64"},
    {Type::I8, "I8"},   {Type::I16, "I16"}, {Type::I32, "I32"}, {Type::I64, "I64"},
    {Type::F16, "F16"}, {Type::F32, "F32"}, {Type::F64, "F64"}};

struct Conv
{
    cudnnTensorDescriptor_t input;
    cudnnTensorDescriptor_t output;
    cudnnFilterDescriptor_t kernel;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnConvolutionDescriptor_t conv;

    Len4 _input;
    Len4 _kernel;
    Len4 _output;
    Len2 _stride;
    Len2 _pad;
    Type _type;

    using perfs_t = std::vector<cudnnConvolutionFwdAlgoPerf_t>;

    perfs_t find_algo()
    {
        perfs_t::value_type algos[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        int count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT, returned;
        auto function = cudnnFindConvolutionForwardAlgorithm;

#if SPYKER_CUDNN_MAJOR >= 7
        if (HeuristicConv) function = cudnnGetConvolutionForwardAlgorithm_v7;
#endif
        CudnnCheck(function(cudnn_static->handle, input, kernel, conv, output, count, &returned, algos));
        return perfs_t(algos, algos + returned);
    }
    Size find_light(const perfs_t &list)
    {
        if (!LightConv) return 0;
        for (Size i = 0; i < list.size(); ++i)
            if (list[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM ||          //
                list[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM ||  //
                list[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT ||                 //
                list[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
                return i;
        return 0;
    }
    void set_algo(cudnnConvolutionFwdAlgo_t algo, Size memory, cudnnMathType_t math)
    {
#if SPYKER_CUDNN_MAJOR >= 7
        CudnnCheck(cudnnSetConvolutionMathType(conv, math));
#endif
        this->algo = algo;
        if (memory <= 0 || memory <= space.size) return;
        space.size = memory;
        if (space.ptr != nullptr) cuda_dealloc(space.ptr);
        space.ptr = cuda_alloc(space.size);
    }
    Size force_memory()
    {
        size_t memory;
        auto algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_static->handle, input, kernel, conv, output, algo, &memory);
        return memory;
    }
    cudnnDataType_t cvt(Type type)
    {
        if (type == Type::F16) return CUDNN_DATA_HALF;
        if (type == Type::F32) return CUDNN_DATA_FLOAT;
        if (type == Type::F64) return CUDNN_DATA_DOUBLE;
        SpykerAssert(false, "CUDA::Conv", "Data type is not supported in DNNL.");
    }
    Conv(Len4 _input, Len4 _kernel, Len4 _output, Len2 _stride, Len2 _pad, Type _type)
        : _input(_input), _kernel(_kernel), _output(_output), _stride(_stride), _pad(_pad), _type(_type)
    {
        if (!cudnn_static) cudnn_static = std::unique_ptr<cudnn>(new cudnn);

        CudnnCheck(cudnnCreateTensorDescriptor(&input));
        CudnnCheck(cudnnCreateTensorDescriptor(&output));
        CudnnCheck(cudnnCreateFilterDescriptor(&kernel));
        CudnnCheck(cudnnCreateConvolutionDescriptor(&conv));

        auto type = cvt(_type);
        CudnnCheck(cudnnSetConvolution2dDescriptor(  //
            conv, _pad.y, _pad.x, _stride.y, _stride.x, 1, 1, CUDNN_CROSS_CORRELATION, type));
        CudnnCheck(cudnnSetTensor4dDescriptor(  //
            input, CUDNN_TENSOR_NCHW, type, _input.t, _input.z, _input.y, _input.x));
        CudnnCheck(cudnnSetFilter4dDescriptor(  //
            kernel, type, CUDNN_TENSOR_NCHW, _kernel.t, _kernel.z, _kernel.y, _kernel.x));
        CudnnCheck(cudnnSetTensor4dDescriptor(  //
            output, CUDNN_TENSOR_NCHW, type, _output.t, _output.z, _output.y, _output.x));

        if (ForceCore)
        {
            set_algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,  //
                     force_memory(), CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
        }
        else
        {
            auto list = find_algo();
            auto index = find_light(list);
            auto algo = list[index].algo;
            auto math = list[index].mathType;

            // if (int(cuda_current_arch()) >= 7 && math == CUDNN_DEFAULT_MATH)
            //     if (algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM ||
            //         algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
            //         math = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
            set_algo(algo, list[index].memory, math);
        }
    }
    ~Conv()
    {
        cudnnDestroyTensorDescriptor(input);
        cudnnDestroyFilterDescriptor(kernel);
        cudnnDestroyTensorDescriptor(output);
        cudnnDestroyConvolutionDescriptor(conv);
    }
    template <typename T>
    void operator()(T *input_, T *kernel_, T *output_)
    {
        F32 alpha = 1, beta = 0;
        CudnnCheck(cudnnConvolutionForward(cudnn_static->handle, &alpha, input, input_, kernel, kernel_,  //
                                           conv, algo, space.ptr, space.size, &beta, output, output_));
    }
    void operator()(F64 *input_, F64 *kernel_, F64 *output_)
    {
        F64 alpha = 1, beta = 0;
        CudnnCheck(cudnnConvolutionForward(cudnn_static->handle, &alpha, input, input_, kernel, kernel_,  //
                                           conv, algo, space.ptr, space.size, &beta, output, output_));
    }
    bool comp(Len4 input_, Len4 kernel_, Len4 output_, Len2 stride_, Len2 pad_, Type type_)
    {
        return _input == input_ && _kernel == kernel_ && _output == output_ &&  //
               _stride == stride_ && _pad == pad_ && _type == type_;
    }
};

std::vector<std::shared_ptr<Conv>> conv_handle;

void conv_clear()
{
    conv_handle.clear();
    if (space.ptr != nullptr) cuda_dealloc(space.ptr);
    space.size = 0, space.ptr = nullptr;
}

Conv &conv_find(Len4 input, Len4 kernel, Len4 output, Len2 stride, Len4 pad, Type type)
{
    auto pad_ = (pad.t != pad.y || pad.z != pad.x) ? Len2{0, 0} : Len2{pad.t, pad.z};
    for (auto conv : conv_handle)
        if (conv->comp(input, kernel, output, stride, pad_, type)) return *conv.get();
    conv_handle.push_back(std::shared_ptr<Conv>(new Conv(input, kernel, output, stride, pad_, type)));
    return *conv_handle.back().get();
}

template <typename T>
void _conv(Vec4<T> input, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    Conv &conv = conv_find(input.len(), kernel.len(), output.len(), stride, pad, TypeName<T>());
    conv(input.data, kernel.data, output.data);
}

#else
template <typename T>
void _conv(Vec4<T> input, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    SpykerAssert(false, "CUDA::Conv", "Conv operation needs cuDNN to work.");
}

void conv_clear() {}
#endif

template <typename T>
void conv(Vec4<T> input_, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    auto input = input_;
    if ((pad.t != pad.y || pad.z != pad.x) && (pad.t != 0 || pad.z != 0 || pad.y != 0 || pad.x != 0))
    {
        input = init<T>(input_.t, input_.z, input_.y + pad.t + pad.y, input_.x + pad.z + pad.x);
        cuda_pad(todyn(Vec3<T>(input_.data, input_.t * input_.z, input_.y, input_.x)),
                 todyn(Vec3<T>(input.data, input.t * input.z, input.y, input.x)), pad, F64(0));
    }
    _conv(input, kernel, output, stride, pad);
    if (input.data != input_.data) deinit(input);
}
}  // namespace CUDA

void cuda_conv(Dyn4 input, Dyn4 kernel, Dyn4 output, Len2 stride, Len4 pad)
{
    IfReal(T, input.type, CUDA::conv<T>(input, kernel, output, stride, pad));
}
void cuda_conv_clear()  //
{
    CUDA::conv_clear();
}
void cuda_conv_options(Size light, Size heuristic, Size force)
{
    if (light > -1) CUDA::LightConv = light;
    if (heuristic > -1) CUDA::HeuristicConv = heuristic;
    if (force > -1) CUDA::ForceCore = force;
}
}  // namespace Core
}  // namespace Spyker
