#include "base.h"
//

namespace Spyker
{
namespace Core
{
namespace CPU
{
template <typename T>
void backward(Size size, PTR(T, input), PTR(T, output), I64 target, Size time, T gamma)
{
    T min = minval(Vec1<T>(output, size));
    T upper = min + gamma, value = std::min(T(time), upper);

    if (min != time)
        for (Size i = 0; i < size; ++i)  //
            output[i] = (output[i] < upper) ? value : output[i];

    output[target] = std::max(T(0), min /*- gamma*/);
    for (Size i = 0; i < size; ++i) output[i] = (output[i] - input[i]) / time;
}
template <typename T>
void backward(Vec2<T> input, Vec2<T> output, Vec1<I64> target, Size time, T gamma)
{
#pragma omp parallel for
    for (Size i = 0; i < input.y; ++i)
    {
        copy(input(i), output(i));
        backward(input.x, input(i).data, output(i).data, target(i), time, gamma);
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void gather(ARG2(T, input), ARG1(U16, output), T threshold)
{
    VEC2(T, input) VEC1(U16, output);
    fill(output, U16(input.y));

    for (Size i = input.y - 1; i >= 0; --i)
    {
        U16 time = i;
        auto id = input(i);
        for (Size j = 0; j < input.x; ++j)
            if (id(j) > threshold) output(j) = time;
    }
}
template <typename T>
void labelize(Vec3<T> input, Vec1<I64> output, T threshold)
{
    BatchSize(input.z);
    auto temp = init<U16>(batch_size, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i)
    {
        auto middle = temp(BatchIndex(i));
        gather(ARG(input(i)), ARG(middle), threshold);
        Size min = minidx(middle);
        if (middle(min) == input.y) min = maxidx(input(i, input.y - 1));
        output(i) = min;
    }

    deinit(temp);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
void sign(Size size, PTR(T, kernel), PTR(T, sign))
{
    for (Size i = 0; i < size; ++i) sign[i] = kernel[i] > 0 ? 1 : (kernel[i] < 0 ? -1 : 0);
}
template <typename T>
void multiply(Size size, PTR(T, input), T output, T grad, PTR(F32, select))
{
    for (Size i = 0; i < size; ++i) select[i] = (input[i] < output) * grad;
}
template <typename T>
void update(Size size, PTR(T, kernel), PTR(T, select), T lrate)
{
    for (Size i = 0; i < size; ++i) kernel[i] -= lrate * select[i];
}
template <typename T>
void reupdate(Size size, PTR(T, kernel), T lrate, T lambda)
{
    T mult = lrate * lambda;
    for (Size i = 0; i < size; ++i) kernel[i] -= mult * kernel[i];
}
template <typename T>
T factor(Size size, PTR(T, sign), PTR(T, select), T lrf)
{
    T sum = 0;
    for (Size i = 0; i < size; ++i) sum += lrf * select[i] * sign[i];
    return sum / size;
}
template <typename T>
void gradient(Size size, PTR(F32, kernel), PTR(F32, select), PTR(T, next))
{
    for (Size i = 0; i < size; ++i) next[i] += kernel[i] * select[i];
}
template <typename T>
void fcbackward(Vec2<F32> kernel, Vec2<T> input, Vec2<T> output, Vec2<T> grad, Vec2<T> next, BPConfig &config)
{
    auto sign = init<F32>(kernel.y, kernel.x);
    auto select = init<F32>(kernel.y, kernel.x);
    CPU::sign(kernel.size(), kernel.data, sign.data);

    for (Size i = 0; i < output.y; ++i)
    {
        for (Size j = 0; j < output.x; ++j)
            multiply(input.x, input(i).data, *output(i, j), *grad(i, j), select(j).data);

        update(kernel.size(), kernel.data, select.data, config.lrate);
        reupdate(kernel.size(), kernel.data, config.lrate, config.lambda);
        config.sfactor -= factor(sign.size(), sign.data, select.data, config.lrf);

        fill(next(i), T(0));
        for (Size j = 0; j < output.x; ++j)  //
            gradient(input.x, kernel(j).data, select(j).data, next(i).data);
    }

    deinit(sign, select);
}
}  // namespace CPU

void cpu_backward(Dyn2 input, Dyn2 output, Dyn1 target, Size time, Scalar gamma)
{
    IfType(T, input.type, CPU::backward<T>(input, output, target, time, gamma));
}
void cpu_labelize(Dyn3 input, Dyn1 output, Scalar threshold)
{
    IfType(T, input.type, CPU::labelize<T>(input, output, threshold));
}
void cpu_fcbackward(Dyn2 kernel, Dyn2 input, Dyn2 output, Dyn2 grad, Dyn2 next, BPConfig &config)
{
    IfType(T, input.type, CPU::fcbackward<T>(kernel, input, output, grad, next, config))
}
}  // namespace Core
}  // namespace Spyker
