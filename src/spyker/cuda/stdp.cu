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
using Configs = std::vector<STDPConfig>;

namespace CUDA
{
template <typename T>
__global__ void rank_fcstdp(Size Y, Size X, PTR(T, input), PTR(U16, output))
{
    input += blockIdx.y * Y * X, output += blockIdx.y * X;
    Cize j = Index1;
    if (X <= j) return;

    U16 value = Y;
    for (Cize i = Y - 1; i >= 0; --i)
        if (input[i * X + j]) value = i;
    output[j] = value;
}

template <typename T>
__global__ void rank_fcstdp(Size Y, Size X, Len2 len, PTR(T, input), PTR(U16, output), PTR(I32, index))
{
    Cize idx = Index1;
    input += (idx / len.x) * Y * X;
    if (len.y * len.x <= idx) return;

    I32 j = index[idx];
    if (j < 0) return;

    U16 value = Y;
    for (Cize i = Y - 1; i >= 0; --i)
        if (input[i * X + j]) value = i;
    output[idx] = value;
}

template <typename K>
__global__ void rank_fcstdp(Size size, PTR(U16, isum), PTR(U16, osum), PTR(K, kernel), STDPConfig config)
{
    Cize idx = Index1;
    if (idx >= size) return;

    F64 old = kernel[idx];
    bool dir = (isum[idx] <= osum[0]);
    F64 value = (config.stabilize ? (old - config.lower) * (config.upper - old) : 1);
    value = old + (dir ? config.positive : config.negative) * value;
    kernel[idx] = max(config.lower, min(value, config.upper));
}

template <typename T>
void rank_fcstdp(Vec3<T> input, Vec2<U16> isum)
{
    rank_fcstdp<<<Config1(1, isum.y, isum.x)>>>(input.y, input.x, input.data, isum.data);
}

template <typename T>
void rank_fcstdp(Vec3<T> output, Vec2<U16> osum, Vec2<I32> index, const Winners &winners)
{
    std::vector<I32> idx(index.size(), -1);
    for (size_t i = 0; i < winners.size(); ++i)
        for (size_t j = 0; j < winners[i].size(); ++j)  //
            idx[i * index.x + j] = winners[i][j].x;

    cpu2cuda(index.size() * sizeof(I32), idx.data(), index.data);
    rank_fcstdp<<<Config1(1, 1, index.size())>>>(output.y, output.x, index.len(), output.data, osum.data, index.data);
}

template <typename K>
void rank_fcstdp(Vec1<U16> isum, U16 *osum, Vec1<K> kernel, STDPConfig stdp)
{
    rank_fcstdp<<<Config1(1, 1, kernel.x)>>>(kernel.x, isum.data, osum, kernel.data, stdp);
}

template <typename T, typename K>
void rank_fcstdp(Vec3<T> input, Vec2<K> kernel, Vec3<T> output, const Configs &configs, const Winners &winners)
{
    Size count = 0;
    for (auto &list : winners) count = std::max<Size>(count, list.size());
    if (count <= 0) return;

    auto isum = init<U16>(input.z, input.x);
    auto osum = init<U16>(input.z, count);
    auto index = init<I32>(input.z, count);

    rank_fcstdp(input, isum);
    rank_fcstdp(output, osum, index, winners);

    for (Size i = 0; i < input.z; ++i)
        for (Size j = 0; j < Size(winners[i].size()); ++j)
        {
            auto winner = winners[i][j];
            if (winner.c < 0) continue;
            SpykerCompare(winner.c, <, configs.size(), "CUDA::STDP", "Config index is out of range.");
            rank_fcstdp(isum(i), osum(i, j), kernel(winner.x), configs[winner.c]);
        }

    deinit(isum, osum, index);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename K>
__global__ void rank_convstdp(Len3 input, PTR(U16, id), PTR(U16, od), Len3 kernel, PTR(K, kd), Len2 start,
                              STDPConfig config)
{
    id += blockIdx.z * input.y * input.x;
    Size x = blockIdx.x * Thread2 + threadIdx.x;
    Size y = blockIdx.y * Thread2 + threadIdx.y;
    if (y >= kernel.y || x >= kernel.x) return;

    kd += (blockIdx.z * kernel.y + y) * kernel.x + x;

    F64 old = kd[0];
    bool dir = (id[(y + start.y) * input.x + (x + start.x)] <= od[0]);
    F64 value = (config.stabilize ? (old - config.lower) * (config.upper - old) : 1);
    value = old + (dir ? config.positive : config.negative) * value;
    kd[0] = max(config.lower, min(value, config.upper));

    //    I32 diff = I32(id[(y + start.y) * input.x + (x + start.x)]) - I32(od[0]);
    //    F32 mult = diff <= 0 ? stdp.pos * expf(diff) : stdp.neg * expf(-diff);
    //    kd[0] += mult * (stdp.stabilize ? (kd[0] - stdp.low) * (stdp.high - kd[0]) : 1);
    //    kd[0] = max(stdp.low, min(kd[0], stdp.high));
}

template <typename T>
void rank_convstdp(Vec5<T> output, Vec2<U16> osum, Vec2<I32> index, const Winners &winners)
{
    std::vector<I32> idx(index.size(), -1);
    for (size_t i = 0; i < winners.size(); ++i)
        for (size_t j = 0; j < winners[i].size(); ++j)
        {
            auto W = winners[i][j];
            idx[i * index.x + j] = (W.z * output.y + W.y) * output.x + W.x;
        }

    cpu2cuda(index.size() * sizeof(I32), idx.data(), index.data);
    Len2 dim = {output.t, output.z * output.y * output.x};
    rank_fcstdp<<<Config1(1, 1, index.size())>>>(dim.y, dim.x, index.len(), output.data, osum.data, index.data);
}

template <typename K>
void rank_convstdp(Vec3<U16> isum, U16 *osum, Vec4<K> kernel_, Len2 stride, Winner winner, STDPConfig stdp)
{
    auto kernel = kernel_(winner.z);
    Len2 start = {winner.y * stride.y, winner.x * stride.x};
    rank_convstdp<<<Config2(kernel.z, kernel.y, kernel.x)>>>(isum.len(), isum.data, osum, kernel.len(), kernel.data,
                                                             start, stdp);
}

template <typename T, typename K>
void rank_convstdp(Vec5<T> input, Vec4<K> kernel, Vec5<T> output, const Configs &configs, const Winners &winners,
                   Len2 stride)
{
    Size count = 0;
    for (auto &list : winners) count = std::max<Size>(count, list.size());
    if (count <= 0) return;

    auto isum = init<U16>(input.u, input.z, input.y, input.x);
    auto osum = init<U16>(input.u, count);
    auto index = init<I32>(input.u, count);

    rank_fcstdp(Vec3<T>{input.data, input.u, input.t, input.z * input.y * input.x},
                Vec2<U16>{isum.data, isum.t, isum.z * isum.y * isum.x});
    rank_convstdp(output, osum, index, winners);

    for (Size i = 0; i < input.u; ++i)
        for (Size j = 0; j < Size(winners[i].size()); ++j)
        {
            auto winner = winners[i][j];
            if (winner.c < 0) continue;
            SpykerCompare(winner.c, <, configs.size(), "CUDA::STDP", "Config index is out of range.");
            rank_convstdp(isum(i), osum(i, j), kernel, stride, winner, configs[winner.c]);
        }

    deinit(isum, osum, index);
}

template <typename T, typename K>
void rank_convstdp(Vec5<T> input_, Vec4<K> kernel, Vec5<T> output, const Configs &configs, const Winners &winners,
                   Len2 stride, Len4 pad)
{
    auto input = input_;
    if (pad.t != 0 || pad.z != 0 || pad.y != 0 || pad.x != 0)
    {
        input = init<T>(input_.u, input_.t, input_.z, input_.y + pad.t + pad.y, input_.x + pad.z + pad.x);
        cuda_pad(todyn(Vec3<T>(input_.data, input_.u * input_.t * input_.z, input_.y, input_.x)),
                 todyn(Vec3<T>(input.data, input.u * input.t * input.z, input.y, input.x)), pad, F64(0));
    }
    rank_convstdp(input, kernel, output, configs, winners, stride);
    if (input.data != input_.data) deinit(input);
}
}  // namespace CUDA

void cuda_rank_fcstdp(Dyn3 input, Dyn2 kernel, Dyn3 output, const Configs &configs, const Winners &winners)
{
    IfType(T, input.type,
           IfReal(K, kernel.type, CUDA::rank_fcstdp<T Comma K>(input, kernel, output, configs, winners)));
}
void cuda_rank_convstdp(Dyn5 input, Dyn4 kernel, Dyn5 output, const Configs &configs, const Winners &winners,
                        Len2 stride, Len4 pad)
{
    IfType(
        T, input.type,
        IfReal(K, kernel.type, CUDA::rank_convstdp<T Comma K>(input, kernel, output, configs, winners, stride, pad)));
}
}  // namespace Core
}  // namespace Spyker
