#include "base.h"
//

namespace Spyker
{
namespace Core
{
namespace CPU
{
#ifdef SPYKER_USE_DNNL
struct Conv
{
    dnnl::convolution_forward conv;
    dnnl::memory::desc input, kernel, output;

    Len4 _input;
    Len4 _kernel;
    Len4 _output;
    Len2 _stride;
    Len4 _pad;
    Type _type;

    dnnl::memory::data_type cvt(Type type)
    {
        if (type == Type::F32) return dnnl::memory::data_type::f32;
        SpykerAssert(false, "CPU::Conv", "Data type " << type << " is not supported with DNNL.");
    }
    Conv(Len4 _input, Len4 _kernel, Len4 _output, Len2 _stride, Len4 _pad, Type _type)
        : _input(_input), _kernel(_kernel), _output(_output), _stride(_stride), _pad(_pad), _type(_type)
    {
        if (!onednn_static) onednn_static = std::unique_ptr<onednn>(new onednn);

        auto type = cvt(_type);
        input = dnnl::memory::desc({_input.t, _input.z, _input.y, _input.x}, type, dnnl::memory::format_tag::nchw);
        kernel = dnnl::memory::desc({_kernel.t, _kernel.z, _kernel.y, _kernel.x}, type, dnnl::memory::format_tag::nchw);
        output = dnnl::memory::desc({_output.t, _output.z, _output.y, _output.x}, type, dnnl::memory::format_tag::nchw);

        dnnl::convolution_forward::desc desc(dnnl::prop_kind::forward_inference,                 //
                                             dnnl::algorithm::convolution_auto,                  //
                                             input, kernel, {}, output, {_stride.y, _stride.x},  //
                                             {_pad.t, _pad.z}, {_pad.y, _pad.x});
        dnnl::convolution_forward::primitive_desc prim(desc, onednn_static->engine);
        conv = dnnl::convolution_forward(prim);
    }
    template <typename T>
    void operator()(T *input_, T *kernel_, T *output_)
    {
        dnnl::memory imem(input, onednn_static->engine, input_);
        dnnl::memory kmem(kernel, onednn_static->engine, kernel_);
        dnnl::memory omem(output, onednn_static->engine, output_);
        conv.execute(onednn_static->stream, {{DNNL_ARG_SRC, imem}, {DNNL_ARG_WEIGHTS, kmem}, {DNNL_ARG_DST, omem}});
        onednn_static->stream.wait();
    }
    bool comp(Len4 input_, Len4 kernel_, Len4 output_, Len2 stride_, Len4 pad_, Type type_)
    {
        return _input == input_ && _kernel == kernel_ && _output == output_ &&  //
               _stride == stride_ && _pad == pad_ && _type == type_;
    }
};

std::vector<std::shared_ptr<Conv>> conv_handle;

Conv &conv_find(Len4 input, Len4 kernel, Len4 output, Len2 stride, Len4 pad, Type type)
{
    for (const auto &conv : conv_handle)
        if (conv->comp(input, kernel, output, stride, pad, type)) return *conv.get();
    conv_handle.push_back(std::shared_ptr<Conv>(new Conv(input, kernel, output, stride, pad, type)));
    return *conv_handle.back().get();
}

template <typename T>
void _conv(Vec4<T> input, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    Conv &conv = conv_find(input.len(), kernel.len(), output.len(), stride, pad, TypeName<T>());
    conv(input.data, kernel.data, output.data);
}
void conv_clear() { conv_handle.clear(); }

#else

#ifdef SPYKER_USE_BLAS
template <typename T>
void transform1(ARG3(T, input), ARG3(T, output), Len2 start)
{
    VEC3(T, input) VEC3(T, output);
    for (Size i = 0; i < output.z; ++i)
        for (Size j = 0; j < output.y; ++j)
        {
            T *id = &input(i, j + start.y, start.x);
            std::copy(id, id + output.x, output(i, j).data);
        }
}

template <typename T>
void transform1(Vec3<T> input, Vec3<T> output, Len2 kernel, Len2 stride)
{
    for (Size j = 0; j < output.z; ++j)
        for (Size k = 0; k < output.y; ++k)
        {
            Len3 len(input.z, kernel.y, kernel.x);
            Len2 start(j * stride.y, k * stride.x);
            transform1(ARG(input), ARG(Vec3<T>(output(j, k).data, len)), start);
        }
}

template <typename T>
void transform2(ARG2(T, input), ARG4(T, output), Len2 kernel, Size stride)
{
    VEC2(T, input) VEC4(T, output);
    for (Size i = 0; i < output.y; ++i)
        for (Size j = 0; j < kernel.y; ++j)
            for (Size k = 0; k < kernel.x; ++k)
            {
                T *id = input(i * stride + j, k);
                std::copy(id, id + output.x, output(j, k, i).data);
            }
}

template <typename T>
void transform2(ARG2(T, input), ARG4(T, output), Len2 kernel, Len2 stride)
{
    VEC2(T, input) VEC4(T, output);
    for (Size i = 0; i < output.y; ++i)
        for (Size j = 0; j < kernel.y; ++j)
            for (Size k = 0; k < kernel.x; ++k)
            {
                T *od = output(j, k, i).data;
                T *id = input(i * stride.y + j, k);
                for (Size t = 0; t < output.x; ++t) od[t] = id[t * stride.x];
            }
}

template <typename T>
void transform2(Vec3<T> input, Vec3<T> output, Len2 kernel, Len2 stride)
{
    for (Size j = 0; j < input.z; ++j)
    {
        Vec4<T> od(output(j * kernel.y * kernel.x).data,  //
                   kernel.y, kernel.x, output.y, output.x);

        if (stride.x == 1)
            transform2(ARG(input(j)), ARG(od), kernel, stride.y);
        else
            transform2(ARG(input(j)), ARG(od), kernel, stride);
    }
}

template <typename T>
void multiply(Vec2<T> input, Vec2<T> kernel, Vec2<T> output, bool trans)
{
    auto kd = Blasw::rmat(kernel.data, kernel.y, kernel.x);
    Blasw::dot(Blasw::rmat(input.data, input.y, input.x), trans ? kd.trans() : kd,  //
               Blasw::rmat(output.data, output.y, output.x), 1, 0);
}

template <typename T>
void conv1(Vec3<T> input_, Vec4<T> kernel, Vec3<T> output, Len2 stride)
{
    auto input = init<T>(output.y, output.x, input_.z * kernel.y * kernel.x);
    transform1(input_, input, {kernel.y, kernel.x}, stride);
    multiply(Vec2<T>{input.data, input.z * input.y, input.x},
             Vec2<T>{kernel.data, kernel.t, kernel.z * kernel.y * kernel.x},
             Vec2<T>{output.data, output.z, output.y * output.x}, true);
    deinit(input);
}

template <typename T>
void conv2(Vec3<T> input_, Vec4<T> kernel, Vec3<T> output, Len2 stride)
{
    auto input = init<T>(input_.z * kernel.y * kernel.x, output.y, output.x);
    transform2(input_, input, {kernel.y, kernel.x}, stride);
    multiply(Vec2<T>{input.data, input.z, input.y * input.x},
             Vec2<T>{kernel.data, kernel.t, kernel.z * kernel.y * kernel.x},
             Vec2<T>{output.data, output.z, output.y * output.x}, false);
    deinit(input);
}

template <typename T>
void _conv(Vec4<T> input_, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    // TODO test
    auto input = input_;
    if (pad.t != 0 || pad.z != 0 || pad.y != 0 || pad.x != 0)
    {
        input = init<T>(input.t, input.z, input.y + pad.t + pad.y, input.x + pad.z + pad.x);
        cpu_pad(todyn(Vec3<T>(input_.data, input_.t * input_.z, input_.y, input_.x)),
                todyn(Vec3<T>(input.data, input.t * input.z, input.y, input.x)), pad, T(0));
    }

    for (Size i = 0; i < input.t; ++i) conv2(input(i), kernel, output(i), stride);
    if (input.data != input_.data) deinit(input);
}
void _conv(Vec4<F16> input_, Vec4<F16> kernel, Vec4<F16> output, Len2 stride, Len4 pad)
{
    SpykerAssert(false, "CPU::Conv", "F16 is not supported with BLAS.");
}
void conv_clear() {}

#else

template <typename T>
void _conv(Vec4<T> input_, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    SpykerCompare(false, ==, true, "CPU::Conv", "BLAS and LAPACK are not enabled in this build.");
}
void conv_clear() {}

#endif

#endif

template <typename T>
void conv(Vec4<T> input, Vec4<T> kernel, Vec4<T> output, Len2 stride, Len4 pad)
{
    _conv(input, kernel, output, stride, pad);
}
}  // namespace CPU

void cpu_conv(Dyn4 input, Dyn4 kernel, Dyn4 output, Len2 stride, Len4 pad)
{
    IfReal(T, kernel.type, CPU::conv<T>(input, kernel, output, stride, pad));
}
void cpu_conv_clear() { CPU::conv_clear(); }
}  // namespace Core
}  // namespace Spyker
