#include "base.h"

namespace Spyker
{
namespace Core
{
namespace CPU
{
#ifdef SPYKER_USE_DNNL
struct FC
{
    dnnl::matmul mult;
    dnnl::memory::desc input, kernel, output;

    Len3 _input;
    Len2 _kernel;
    Len3 _output;
    Type _type;

    dnnl::memory::data_type cvt(Type type)
    {
        if (type == Type::F16) return dnnl::memory::data_type::f16;
        if (type == Type::F32) return dnnl::memory::data_type::f32;
        SpykerAssert(false, "CPU::FC", "Data type is not supported in DNNL.");
    }
    FC(Len3 _input, Len2 _kernel, Len3 _output, Type _type)
        : _input(_input), _kernel(_kernel), _output(_output), _type(_type)
    {
        if (!onednn_static) onednn_static = std::unique_ptr<onednn>(new onednn);

        auto type = cvt(_type);
        input = dnnl::memory::desc({_input.z * _input.y, _input.x}, type, dnnl::memory::format_tag::ab);
        kernel = dnnl::memory::desc({_kernel.x, _kernel.y}, type, dnnl::memory::format_tag::ba);
        output = dnnl::memory::desc({_output.z * _output.y, _output.x}, type, dnnl::memory::format_tag::ab);

        dnnl::matmul::desc desc(input, kernel, output);
        dnnl::matmul::primitive_desc prim(desc, onednn_static->engine);
        mult = dnnl::matmul(prim);
    }
    template <typename T>
    void operator()(T *input_, T *kernel_, T *output_)
    {
        dnnl::memory imem(input, onednn_static->engine, input_);
        dnnl::memory kmem(kernel, onednn_static->engine, kernel_);
        dnnl::memory omem(output, onednn_static->engine, output_);
        mult.execute(onednn_static->stream, {{DNNL_ARG_SRC, imem}, {DNNL_ARG_WEIGHTS, kmem}, {DNNL_ARG_DST, omem}});
        onednn_static->stream.wait();
    }
    bool comp(Len3 input_, Len2 kernel_, Len3 output_, Type type_)
    {
        return _input == input_ && _kernel == kernel_ && _output == output_ && _type == type_;
    }
};

std::vector<std::shared_ptr<FC>> fc_handle;

void fc_clear() { fc_handle.clear(); }

FC &fc_find(Len3 input, Len2 kernel, Len3 output, Type type)
{
    for (const auto &fc : fc_handle)
        if (fc->comp(input, kernel, output, type)) return *fc.get();
    fc_handle.push_back(std::shared_ptr<FC>(new FC(input, kernel, output, type)));
    return *fc_handle.back().get();
}

template <typename T>
void _fc(Vec3<T> input, Vec2<T> kernel, Vec3<T> output)
{
    fc_find(input.len(), kernel.len(), output.len(), TypeName<T>())(input.data, kernel.data, output.data);
}

#ifdef SPYKER_USE_BLAS
void _fc(Vec3<F64> input, Vec2<F64> kernel, Vec3<F64> output)
{
    Blasw::dot(Blasw::rmat(input.data, input.z * input.y, input.x),
               Blasw::rmat(kernel.data, kernel.y, kernel.x).trans(),
               Blasw::rmat(output.data, output.z * output.y, output.x), 1, 0);
}
#else
void _fc(Vec3<F64> input, Vec2<F64> kernel, Vec3<F64> output)
{
    SpykerAssert(false, "CPU::FC", "F64 is not supported with DNNL.");
}
#endif

#else

#ifdef SPYKER_USE_BLAS
template <typename T>
void _fc(Vec3<T> input, Vec2<T> kernel, Vec3<T> output)
{
    Blasw::dot(Blasw::rmat(input.data, input.z * input.y, input.x),
               Blasw::rmat(kernel.data, kernel.y, kernel.x).trans(),
               Blasw::rmat(output.data, output.z * output.y, output.x), 1, 0);
}
void _fc(Vec3<F16> input, Vec2<F16> kernel, Vec3<F16> output)
{
    SpykerAssert(false, "CPU::FC", "F16 is not supported with BLAS.");
}

#else

template <typename T>
void _fc(Vec3<T> input, Vec2<T> kernel, Vec3<T> output)
{
    SpykerAssert(false, "CPU::FC", "FC operation needs DNNL or BLAS to work.");
}
#endif

void fc_clear() {}
#endif

template <typename T>
void fc(Vec3<T> input, Vec2<T> kernel, Vec3<T> output)
{
    _fc(input, kernel, output);
}

template <typename T>
void sign(Size size, PTR(T, input), PTR(T, output))
{
#pragma omp parallel for
    for (Size i = 0; i < size; ++i)  //
        output[i] = T(input[i] > 0 ? 1 : (input[i] < 0 ? -1 : 0));
}

template <typename T>
void signfc(Vec3<T> input, Vec2<T> kernel, Vec3<T> output)
{
    auto temp = init<T>(kernel.y, kernel.x);
    sign(kernel.size(), kernel.data, temp.data);
    fc(input, temp, output);
    deinit(temp);
}
}  // namespace CPU

void cpu_fc(Dyn3 input, Dyn2 kernel, Dyn3 output)  //
{
    IfReal(T, kernel.type, CPU::fc<T>(input, kernel, output));
}
void cpu_signfc(Dyn3 input, Dyn2 kernel, Dyn3 output)  //
{
    IfReal(T, kernel.type, CPU::signfc<T>(input, kernel, output));
}
void cpu_fc_clear() { CPU::fc_clear(); }
}  // namespace Core
}  // namespace Spyker
