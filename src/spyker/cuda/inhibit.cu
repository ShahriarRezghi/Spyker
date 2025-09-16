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
namespace CUDA
{
template <typename T>
__global__ void rank_inhibit(Cize Y, Cize X, PTR(T, input), PTR(T, maximum), PTR(U16, channel))
{
    input += blockIdx.y * Y * X, maximum += blockIdx.y * X, channel += blockIdx.y * X;
    Cize j = Index1;
    if (X <= j) return;

    U16 index = 0;
    T value = input[j];
    for (Cize i = 1; i < Y; ++i)
        if (input[i * X + j] > value) value = input[i * X + j], index = i;
    maximum[j] = value, channel[j] = index;
}

template <typename T>
__global__ void rank_inhibit(Cize Y, Cize X, PTR(T, maximum), PTR(U16, sum), T thresh)
{
    maximum += blockIdx.y * Y * X, sum += blockIdx.y * X;
    Cize j = Index1;
    if (X <= j) return;

    U16 value = 0;
    for (Cize i = 0; i < Y; ++i) value += (maximum[i * X + j] > thresh);
    sum[j] = value;
}

__global__ void rank_inhibit(Cize Y, Cize X, PTR(U16, channel), PTR(U16, sum), PTR(U16, index))
{
    channel += blockIdx.y * Y * X, sum += blockIdx.y * X, index += blockIdx.y * X;
    Cize i = Index1;
    if (X <= i) return;

    U16 time = Y - max(sum[i], U16(1));
    index[i] = channel[time * X + i];
}

template <typename T>
__global__ void rank_inhibit(Cize Z, Cize Y, Cize X, PTR(T, input), PTR(U16, index))
{
    input += (blockIdx.z * Z + blockIdx.y) * Y * X, index += blockIdx.z * X;
    Cize j = Index1;
    if (X <= j) return;
    for (U16 i = 0; i < Y; ++i) input[i * X + j] *= T(index[j] == i);
}

template <typename T>
void rank_inhibit(Vec4<T> input, T threshold)
{
    auto maximum = init<T>(input.t, input.z, input.x);
    auto channel = init<U16>(input.t, input.z, input.x);
    auto sum = init<U16>(input.t, input.x);
    auto index = init<U16>(input.t, input.x);

    rank_inhibit<<<Config1(1, input.t * input.z, input.x)>>>(input.y, input.x, input.data, maximum.data, channel.data);
    rank_inhibit<<<Config1(1, maximum.z, maximum.x)>>>(maximum.y, maximum.x, maximum.data, sum.data, threshold);
    rank_inhibit<<<Config1(1, channel.z, channel.x)>>>(channel.y, channel.x, channel.data, sum.data, index.data);
    rank_inhibit<<<Config1(input.t, input.z, input.x)>>>(input.z, input.y, input.x, input.data, index.data);

    deinit(maximum, channel, sum, index);
}
}  // namespace CUDA

void cuda_rank_inhibit(Dyn4 input, Scalar threshold) { IfType(T, input.type, CUDA::rank_inhibit<T>(input, threshold)); }
}  // namespace Core
}  // namespace Spyker
