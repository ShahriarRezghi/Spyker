#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
bool LightConv = false;

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

        int index = 0, count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT, returned;
        cudnnConvolutionFwdAlgoPerf_t algos[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];

#if SPYKER_CUDNN_VERSION >= 7
        CudnnCheck(cudnnGetConvolutionForwardAlgorithm_v7(  //
            cudnn_static->handle, input, kernel, conv, output, count, &returned, algos));
#else
        CudnnCheck(cudnnFindConvolutionForwardAlgorithm(  //
            cudnn_static->handle, input, kernel, conv, output, count, &returned, algos));
#endif

        if (LightConv)
            for (Size i = 0; i < returned; ++i)
                if (algos[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM ||          //
                    algos[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM ||  //
                    algos[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT ||                 //
                    algos[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
                {
                    index = i;
                    break;
                }

#if SPYKER_CUDNN_VERSION >= 7
        CudnnCheck(cudnnSetConvolutionMathType(conv, algos[index].mathType));
#endif

        algo = algos[index].algo;
        if (algos[index].memory != 0 && space.size < algos[index].memory)
        {
            space.size = algos[index].memory;
            if (space.ptr != nullptr) cuda_dealloc(space.ptr);
            space.ptr = cuda_alloc(space.size);
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

void light_conv(bool light)
{
    SpykerCompare(space.ptr, ==, (void *)nullptr, "CUDA:Conv",
                  "Light convolution can't be set after using convolutional layers.");
    LightConv = light;
}

#else
template <typename T>
void _conv(Vec4<T> input, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    SpykerAssert(false, "CUDA::Conv", "Conv operation needs cuDNN to work.");
}

void conv_clear() {}

void light_conv(bool light) { LightConv = light; }
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
void cuda_conv_clear() { CUDA::conv_clear(); }
void cuda_light_conv(bool light) { CUDA::light_conv(light); }
}  // namespace Core
}  // namespace Spyker
