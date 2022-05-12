#include "base.h"
//

namespace Spyker
{
namespace Core
{
namespace CPU
{
template <typename T>
void dog(ARG1(T, input1), ARG1(T, input2), ARG1(T, output))
{
    VEC1(T, input1) VEC1(T, input2) VEC1(T, output);
    for (Size i = 0; i < output.x; ++i) output(i) = std::max(input1(i) - input2(i), T(0));
}

template <typename I>
void dog(Vec4<I> input, Vec4<F32> kernel, Vec4<F32> output_, Len4 pad)
{
    auto middle_ = init<F32>(output_.t, kernel.t, output_.y, output_.x);
    cpu_conv(todyn(input), todyn(kernel), todyn(middle_), {1, 1}, pad);
    Vec3<F32> middle(middle_.data, middle_.t, 2, middle_.z / 2 * middle_.y * middle_.x);
    Vec2<F32> output(output_.data, output_.t, output_.z * output_.y * output_.x);

#pragma omp parallel for
    for (Size i = 0; i < output.y; ++i)  //
        dog(ARG(middle(i, 0)), ARG(middle(i, 1)), ARG(output(i)));

    deinit(middle);
}

template <typename T>
void log(ARG1(T, input1), ARG1(T, input2))
{
    VEC1(T, input1) VEC1(T, input2);

    for (Size i = 0; i < input1.x; ++i)
    {
        T diff = input1(i) - input2(i);
        input1(i) = std::max(diff, T(0));
        input2(i) = std::max(-diff, T(0));
    }
}

template <typename T>
void log(Vec4<T> input, Vec4<F32> kernel, Vec4<F32> output_, Len4 pad)
{
    cpu_conv(todyn(input), todyn(kernel), todyn(output_), {1, 1}, pad);
    Vec3<F32> output(output_.data, output_.t, 2, output_.z / 2 * output_.y * output_.x);

#pragma omp parallel for
    for (Size i = 0; i < output.z; ++i) log(ARG(output(i, 0)), ARG(output(i, 1)));
}
}  // namespace CPU

void cpu_dog(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad)
{
    IfType(T, input.type, CPU::dog<T>(input, kernel, output, pad));
}
void cpu_log(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad)
{
    IfType(T, input.type, CPU::log<T>(input, kernel, output, pad));
}
void cpu_gabor(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad) { cpu_conv(input, kernel, output, {1, 1}, pad); }
}  // namespace Core
}  // namespace Spyker
