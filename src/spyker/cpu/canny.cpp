#include "base.h"
//

namespace Spyker
{
namespace Core
{
namespace CPU
{
template <typename T>
void normalize(PTR(T, input), Size size)
{
    F32 max = 255 / (maxval(Vec1<T>(input, size)) - Epsilon);
    for (Size i = 0; i < size; ++i) input[i] *= max;
}

template <typename T>
void step1(Vec3<T> input_, Vec3<F32> output_)
{
    F32 kernel_[] = {
        0.0030, 0.0133, 0.0219, 0.0133, 0.0030,  //
        0.0133, 0.0596, 0.0983, 0.0596, 0.0133,  //
        0.0219, 0.0983, 0.1621, 0.0983, 0.0219,  //
        0.0133, 0.0596, 0.0983, 0.0596, 0.0133,  //
        0.0030, 0.0133, 0.0219, 0.0133, 0.0030,  //
    };

    Vec4<F32> kernel = {kernel_, 1, 1, 5, 5};
    Vec4<T> input = {input_.data, input_.z, 1, input_.y, input_.x};
    Vec4<F32> output = {output_.data, output_.z, 1, output_.y, output_.x};
    cpu_conv(todyn(input), todyn(kernel), todyn(output), {1, 1}, {2, 2, 2, 2});
}

void step2(Vec3<F32> input_, Vec4<F32> output)
{
    F32 kernel_[] = {
        -1, -1, 0,  1,  1,  //
        -2, -2, 0,  2,  2,  //
        -3, -6, 0,  6,  3,  //
        -2, -2, 0,  2,  2,  //
        -1, -1, 0,  1,  1,  //

        1,  2,  3,  2,  1,   //
        1,  2,  6,  2,  1,   //
        0,  0,  0,  0,  0,   //
        -1, -2, -6, -2, -1,  //
        -1, -2, -3, -2, -1,  //

    };

    Vec4<F32> kernel = {kernel_, 2, 1, 5, 5};
    Vec4<F32> input = {input_.data, input_.z, 1, input_.y, input_.x};
    cpu_conv(todyn(input), todyn(kernel), todyn(output), {1, 1}, {2, 2, 2, 2});
}

void step3_(Size size, PTR(F32, x), PTR(F32, y), PTR(F32, output))
{
    for (Size i = 0; i < size; ++i)  //
        output[i] = std::sqrt(x[i] * x[i] + y[i] * y[i]);
    normalize(output, size);
}

void step3(Size size, PTR(F32, x), PTR(F32, y), PTR(F32, output))
{
    for (Size i = 0; i < size; ++i)  //
        output[i] = std::atan2(y[i], x[i]);
}

void step3(Vec3<F32> input, Vec3<F32> output)
{
    step3_(input.y * input.x, input(0, 0, 0), input(1, 0, 0), output(0, 0, 0));
    step3(input.y * input.x, input(0, 0, 0), input(1, 0, 0), output(1, 0, 0));
}

U8 step4(Vec2<F32> input, F32 angle, Size i, Size j)
{
    angle *= 180 / PI;
    if (angle < 0) angle += 180;
    F32 v = *input(i, j), v1, v2;

    if (angle < 22.5 || 157.5 <= angle)
    {
        v1 = *input(i, j + 1);
        v2 = *input(i, j - 1);
    }
    else if (angle < 67.5)
    {
        v1 = *input(i + 1, j - 1);
        v2 = *input(i - 1, j + 1);
    }
    else if (angle < 112.5)
    {
        v1 = *input(i + 1, j);
        v2 = *input(i - 1, j);
    }
    else
    {
        v1 = *input(i - 1, j - 1);
        v2 = *input(i + 1, j + 1);
    }
    return v >= v1 && v >= v2 ? v : 0;
}

void step4(ARG2(F32, input), ARG2(F32, angle), ARG2(U8, output))
{
    VEC2(F32, input) VEC2(F32, angle) VEC2(U8, output);

    fill(output, U8(0));
    for (Size i = 1; i < input.y - 1; ++i)
    {
        F32 *ad = angle(i, 0);
        U8 *od = output(i, 0);
        for (Size j = 1; j < input.x - 1; ++j) od[j] = step4(input, ad[j], i, j);
    }
    normalize(output.data, output.size());
}

void step4(Vec3<F32> input, Vec2<U8> output) { step4(ARG(input(0)), ARG(input(1)), ARG(output)); }

#define WEAK 1
#define STRONG 2

void step5(Size size, PTR(U8, input), PTR(U8, output), U8 low, U8 high)
{
    for (Size i = 0; i < size; ++i) output[i] = high <= input[i] ? STRONG : (low <= input[i] ? WEAK : 0);
}

void step5(Vec2<U8> input, Vec2<U8> output, U8 low, U8 high)
{
    step5(input.size(), input.data, output.data, low, high);
}

void step6(ARG2(U8, input), ARG2(U8, output))
{
    VEC2(U8, input) VEC2(U8, output);

    copy(input, output);

    for (Size i = -1; i <= 1; ++i)
        for (Size j = -1; j <= 1; ++j)
        {
            if (i == 0 && j == 0) continue;

            for (Size k = 1; k < input.y - 1; ++k)
            {
                U8 *id = input(k + i, j);
                U8 *od = output(k, 0);

#pragma clang loop vectorize(enable)
                for (Size t = 1; t < input.x - 1; ++t)  //
                    if (od[t] == WEAK && id[t] == STRONG) od[t] = STRONG;
            }
        }
}

void step6(Vec2<U8> input, Vec2<U8> output) { step6(ARG(input), ARG(output)); }

void step7(Size size, PTR(U8, input), PTR(U8, output))
{
    for (Size i = 0; i < size; ++i)
    {
        U8 temp = (output[i] == STRONG ? 255 : 0);
        output[i] = std::max(temp, input[i]);
    }
}

void step7(Vec2<U8> input, Vec2<U8> output) { step7(input.size(), input.data, output.data); }

template <typename T>
void canny(Vec3<T> input, Vec3<U8> output, U8 low, U8 high)
{
    auto temp2 = init<F32>(input.z, 2, input.y, input.x);
    auto temp3 = init<F32>(input.z, 2, input.y, input.x);
    Vec3<F32> temp1 = {temp3.data, input.z, input.y, input.x};
    Vec3<U8> temp4 = {(U8 *)temp2.data, input.z, input.y, input.x};
    Vec3<U8> temp5 = {(U8 *)temp3.data, input.z, input.y, input.x};

    step1(input, temp1);
    step2(temp1, temp2);

#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i)
    {
        step3(temp2(i), temp3(i));
        step4(temp3(i), temp4(i));
        step5(temp4(i), temp5(i), low, high);
        step6(temp5(i), output(i));
        step7(temp4(i), output(i));
    }
    deinit(temp2, temp3);
}
}  // namespace CPU

void cpu_canny(Dyn3 input, Dyn3 output, Scalar low, Scalar high)
{
    IfType(T, input.type, CPU::canny<T>(input, output, low, high));
}
}  // namespace Core
}  // namespace Spyker
