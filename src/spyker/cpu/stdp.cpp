#include "base.h"
//

namespace Spyker
{
namespace Core
{
using Configs = std::vector<STDPConfig>;

namespace CPU
{
template <typename T>
void rank_fcstdp(ARG2(T, input), ARG1(U16, sum))
{
    VEC2(T, input) VEC1(U16, sum);

    fill(sum, U16(input.y));
    for (Size i = input.y - 1; i >= 0; --i)
    {
        U16 time = i;
        T *id = input(i, 0);
        for (Size j = 0; j < input.x; ++j)
            if (id[j]) sum(j) = time;
    }
}

template <typename T>
U16 rank_fcstdp(Vec2<T> output, Winner winner)
{
    U16 value = output.y;
    for (Size i = output.y - 1; i >= 0; --i)
        if (output(i, winner.x)) value = i;
    return value;
}

void rank_fcstdp(ARG1(U16, isum), U16 osum, ARG1(F32, kernel), STDPConfig stdp)
{
    VEC1(U16, isum) VEC1(F32, kernel);

    F32 *kd = kernel.data;
    for (Size i = 0; i < kernel.x; ++i)
    {
        F32 mult = (stdp.stabilize ? (kd[i] - stdp.low) * (stdp.high - kd[i]) : 1);
        kd[i] += (isum(i) <= osum ? stdp.pos : stdp.neg) * mult;
        kd[i] = std::max(stdp.low, std::min(kd[i], stdp.high));
    }
}

template <typename T>
void rank_fcstdp(Vec3<T> input, Vec2<F32> kernel, Vec3<T> output, const Configs &configs, const Winners &winners)
{
    auto sum = init<U16>(input.x);
    for (Size i = 0; i < input.z; ++i)
    {
        rank_fcstdp(ARG(input(i)), ARG(sum));
        for (auto winner : winners[i])
        {
            if (winner.c < 0) continue;
            SpykerCompare(winner.c, <, configs.size(), "CPU::STDP", "STDP configuration index is out of range.");
            U16 osum = rank_fcstdp(output(i), winner);
            rank_fcstdp(ARG(sum), osum, ARG(kernel(winner.x)), configs[winner.c]);
        }
    }
    deinit(sum);
}

template <typename T>
U16 rank_convstdp(ARG4(T, input), ARG3(U16, isum), ARG4(T, output), Len2 stride, Winner winner)
{
    VEC4(T, input) VEC3(U16, isum) VEC4(T, output);

    fill(isum, U16(input.t));
    for (Size i = input.t - 1; i >= 0; --i)
        for (Size j = 0; j < input.z; ++j)
            for (Size k = 0; k < isum.y; ++k)
            {
                U16 *td = isum(j, k, 0);
                T *id = input(i, j, winner.y * stride.y + k, winner.x * stride.x);
                for (Size t = 0; t < isum.x; ++t)
                    if (id[t]) td[t] = i;
            }

    U16 od = output.t;
    for (Size i = output.t - 1; i >= 0; --i)
        if (*output(i, winner.z, winner.y, winner.x)) od = i;
    return od;
}

void rank_convstdp(ARG3(U16, sum), U16 osum, ARG3(F32, kernel), STDPConfig stdp)
{
    VEC3(U16, sum) VEC3(F32, kernel);

    U16 *isum = sum.data;
    F32 *kd = kernel.data;
    Size size = kernel.size();

    for (Size i = 0; i < size; ++i)
    {
        F32 mult = (stdp.stabilize ? (kd[i] - stdp.low) * (stdp.high - kd[i]) : 1);
        kd[i] += (isum[i] <= osum ? stdp.pos : stdp.neg) * mult;
        kd[i] = std::max(stdp.low, std::min(kd[i], stdp.high));
    }
}

template <typename T>
void rank_convstdp(Vec5<T> input, Vec4<F32> kernel, Vec5<T> output, const Configs &configs, const Winners &winners,
                   Len2 stride)
{
    auto sum = init<U16>(kernel.z, kernel.y, kernel.x);

    for (Size i = 0; i < input.u; ++i)
        for (auto winner : winners[i])
        {
            if (winner.c < 0) continue;
            SpykerCompare(winner.c, <, configs.size(), "CPU::STDP", "STDP configuration index is out of range.");
            U16 osum = rank_convstdp(ARG(input(i)), ARG(sum), ARG(output(i)), stride, winner);
            rank_convstdp(ARG(sum), osum, ARG(kernel(winner.z)), configs[winner.c]);
        }
    deinit(sum);
}

template <typename T>
void rank_convstdp(Vec5<T> input_, Vec4<F32> kernel, Vec5<T> output, const Configs &configs, const Winners &winners,
                   Len2 stride, Len4 pad)
{
    auto input = input_;
    if (pad.t != 0 || pad.z != 0 || pad.y != 0 || pad.x != 0)
    {
        input = init<T>(input_.u, input_.t, input_.z, input_.y + pad.t + pad.y, input_.x + pad.z + pad.x);
        cpu_pad(todyn(Vec3<T>(input_.data, input_.u * input_.t * input_.z, input_.y, input_.x)),
                todyn(Vec3<T>(input.data, input.u * input.t * input.z, input.y, input.x)), pad, T(0));
    }
    rank_convstdp(input, kernel, output, configs, winners, stride);
    if (input.data != input_.data) deinit(input);
}
}  // namespace CPU

void cpu_rank_fcstdp(Dyn3 input, Dyn2 kernel, Dyn3 output, const Configs &configs, const Winners &winners)
{
    IfType(T, input.type, CPU::rank_fcstdp<T>(input, kernel, output, configs, winners));
}
void cpu_rank_convstdp(Dyn5 input, Dyn4 kernel, Dyn5 output, const Configs &configs, const Winners &winners,
                       Len2 stride, Len4 pad)
{
    IfType(T, input.type, CPU::rank_convstdp<T>(input, kernel, output, configs, winners, stride, pad));
}
}  // namespace Core
}  // namespace Spyker
