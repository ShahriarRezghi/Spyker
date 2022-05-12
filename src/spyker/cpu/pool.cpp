#include "base.h"
//
#include <limits>

namespace Spyker
{
namespace Core
{
namespace CPU
{
template <typename T>
void pad(Vec3<T> input, Vec3<T> output, Len4 pad, T value)
{
#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i)
    {
        fill(output(i), value);
        for (Size j = 0; j < input.y; ++j)  //
            copy(input(i, j), Vec1<T>(output(i, j + pad.t, pad.z), input.x));
    }
}

template <typename T>
void rank_pool(ARG2(T, input), ARG2(T, output), Len2 kernel, Len2 stride)
{
    VEC2(T, input) VEC2(T, output);

    for (Size i = 0; i < output.y; ++i)
        for (Size k = 0; k < kernel.y; ++k)
            for (Size t = 0; t < kernel.x; ++t)
            {
                T *id = input(i * stride.y + k, t);
                T *od = output(i, 0);
#pragma clang loop vectorize(enable) interleave(enable)
                for (Size j = 0; j < output.x; ++j) od[j] = std::max(od[j], id[j * stride.x]);
            }
}

template <typename T>
void rank_pool(Vec3<T> input_, Vec3<T> output, Len2 kernel, Len2 stride, Len4 pad)
{
    auto input = input_;
    if (pad.t != 0 || pad.z != 0 || pad.y != 0 || pad.x != 0)
    {
        input = init<T>(input.z, input.y + pad.t + pad.y, input.x + pad.z + pad.x);
        CPU::pad(input_, input, pad, T(0));
    }

#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i)
    {
        fill(output(i), Limits<T>::min());
        rank_pool(ARG(input(i)), ARG(output(i)), kernel, stride);
    }
    if (input.data != input_.data) deinit(input);
}

template <typename T>
void pool_index(ARG2(T, input), ARG2(I32, index), ARG2(T, value), Len2 kernel, Len2 stride)
{
    VEC2(T, input) VEC2(I32, index) VEC2(T, value);

    for (Size i = 0; i < index.y; ++i)
        for (Size k = 0; k < kernel.y; ++k)
            for (Size t = 0; t < kernel.x; ++t)
            {
                T *id = input(i * stride.y + k, t);
                I32 *xd = index(i, 0);
                T *vd = value(i, 0);

                Size offset = (i * stride.y + k) * input.x + t;
                for (Size j = 0; j < index.x; ++j)
                    if (vd[j] < id[j * stride.x]) vd[j] = id[j * stride.x], xd[j] = offset + j * stride.x;
            }
}

template <typename T>
void rate_pool(Size size, PTR(T, input), PTR(I32, index), PTR(T, output))
{
    for (Size i = 0; i < size; ++i) output[i] = input[index[i]];
}

template <typename T, typename R>
void rate_pool(Vec5<T> input, Vec4<R> rates, Vec5<T> output, Len2 kernel, Len2 stride, Len4 pad)
{
    SpykerAssert(pad.t == 0 && pad.z == 0 && pad.y == 0 && pad.x == 0, "CPU::Pool",
                 "Padding is not supported in rate pooling.");

    BatchSize(output.u);
    auto index = init<I32>(batch_size, output.z, output.y, output.x);
    auto value = init<R>(batch_size, output.z, output.y, output.x);

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
    {
        Size bidx = BatchIndex(i);
        fill(index(bidx), I32(0)), fill(value(bidx), Limits<R>::min());

        for (Size k = 0; k < input.z; ++k)
            pool_index(ARG(rates(i, k)), ARG(index(bidx, k)), ARG(value(bidx, k)), kernel, stride);

        for (Size j = 0; j < input.t; ++j)
            for (Size k = 0; k < input.z; ++k)
                rate_pool(output.y * output.x, input(i, j, k).data, index(bidx, k).data, output(i, j, k).data);
    }
    deinit(index, value);
}
}  // namespace CPU

void cpu_pad(Dyn3 input, Dyn3 output, Len4 pad, Scalar value)
{
    IfType(T, input.type, CPU::pad<T>(input, output, pad, value));
}
void cpu_rank_pool(Dyn4 input_, Dyn4 output_, Len2 kernel, Len2 stride, Len4 pad)
{
    Dyn3 input(input_.data, input_.type, input_.t * input_.z, input_.y, input_.x);
    Dyn3 output(output_.data, output_.type, output_.t * output_.z, output_.y, output_.x);
    IfType(T, input.type, CPU::rank_pool<T>(input, output, kernel, stride, pad));
}
void cpu_rate_pool(Dyn5 input, Dyn4 rates, Dyn5 output, Len2 kernel, Len2 stride, Len4 pad)
{
    IfType(T, input.type, IfType(R, rates.type, CPU::rate_pool<T Comma R>(input, rates, output, kernel, stride, pad)));
}
}  // namespace Core
}  // namespace Spyker

// template <typename T, typename R>
// void rate_pool(Vec3<T> input_, Vec3<R> rates_, Vec3<T> output, Len2 kernel, Len2 stride, Len4 pad)
//{
//     auto index = init<I32>(output.z, output.y, output.x);
//     auto value = init<T>(output.z, output.y, output.x);
//     auto input = input_;
//     auto rates = rates_;

//    if (pad.t != 0 || pad.z != 0 || pad.y != 0 || pad.x != 0)
//    {
//        input = init<T>(input.z, input.y + pad.t + pad.y, input.x + pad.z + pad.x);
//        rates = init<T>(rates.z, rates.y + pad.t + pad.y, rates.x + pad.z + pad.x);
//        CPU::pad(input_, input, pad, T(0)), CPU::pad(rates_, rates, pad, R(0));
//    }

//#pragma omp parallel for
//    for (Size i = 0; i < input.z; ++i)
//    {
//        fill(index, I32(-1)), fill(value, Limits<T>::min());
//        pool_index(input(i), index(i), value(i), kernel, stride);
//        rate_pool(index.y * index.x, input(i).data, index(i).data, output(i).data);
//    }

//    deinit(index, value);
//    if (input.data != input_.data) deinit(input);
//    if (rates.data != rates_.data) deinit(rates);
//}
