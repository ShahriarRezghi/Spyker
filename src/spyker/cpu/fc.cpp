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

    FC(Len3 _input, Len2 _kernel, Len3 _output) : _input(_input), _kernel(_kernel), _output(_output)
    {
        if (!onednn_static) onednn_static = std::unique_ptr<onednn>(new onednn);

        input = dnnl::memory::desc({_input.z * _input.y, _input.x},  //
                                   dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
        kernel = dnnl::memory::desc({_kernel.x, _kernel.y},  //
                                    dnnl::memory::data_type::f32, dnnl::memory::format_tag::ba);
        output = dnnl::memory::desc({_output.z * _output.y, _output.x},  //
                                    dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);

        dnnl::matmul::desc desc(input, kernel, output);
        dnnl::matmul::primitive_desc prim(desc, onednn_static->engine);
        mult = dnnl::matmul(prim);
    }
    void operator()(F32 *input_, F32 *kernel_, F32 *output_)
    {
        dnnl::memory imem(input, onednn_static->engine, input_);
        dnnl::memory kmem(kernel, onednn_static->engine, kernel_);
        dnnl::memory omem(output, onednn_static->engine, output_);
        mult.execute(onednn_static->stream, {{DNNL_ARG_SRC, imem}, {DNNL_ARG_WEIGHTS, kmem}, {DNNL_ARG_DST, omem}});
        onednn_static->stream.wait();
    }
    bool comp(Len3 input_, Len2 kernel_, Len3 output_)
    {
        return _input == input_ && _kernel == kernel_ && _output == output_;
    }
};

std::vector<std::shared_ptr<FC>> fc_handle;

void fc_clear() { fc_handle.clear(); }

FC &fc_find(Len3 input, Len2 kernel, Len3 output)
{
    for (const auto &fc : fc_handle)
        if (fc->comp(input, kernel, output)) return *fc.get();
    fc_handle.push_back(std::shared_ptr<FC>(new FC(input, kernel, output)));
    return *fc_handle.back().get();
}

void fc(Vec3<F32> input, Vec2<F32> kernel, Vec3<F32> output)
{
    fc_find(input.len(), kernel.len(), output.len())(input.data, kernel.data, output.data);
}

#else

#ifdef SPYKER_USE_BLAS
void fc(Vec3<F32> input, Vec2<F32> kernel, Vec3<F32> output)
{
    Blasw::dot(Blasw::rmat(input.data, input.z * input.y, input.x),
               Blasw::rmat(kernel.data, kernel.y, kernel.x).trans(),
               Blasw::rmat(output.data, output.z * output.y, output.x), 1, 0);
}

#else

void fc(Vec3<F32> input, Vec2<F32> kernel, Vec3<F32> output)
{
    SpykerAssert(false, "CPU::FC", "FC operation needs DNNL to work.");
}
#endif

void fc_clear() {}
#endif

template <typename T>
void fc(Vec3<T> input, Vec2<F32> kernel, Vec3<F32> output)
{
    auto temp = init<F32>(input.z, input.y, input.x);
    copy(input, temp);
    fc(temp, kernel, output);
    deinit(temp);
}
}  // namespace CPU

void cpu_fc(Dyn3 input, Dyn2 kernel, Dyn3 output)  //
{
    IfType(T, input.type, CPU::fc<T>(input, kernel, output));
}
void cpu_fc_clear() { CPU::fc_clear(); }
}  // namespace Core
}  // namespace Spyker
