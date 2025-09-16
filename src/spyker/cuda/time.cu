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
template <typename I, typename O>
__global__ void rank_gather(Cize Y, Cize X, PTR(I, input), PTR(O, output), I thresh)
{
    input += blockIdx.y * Y * X, output += blockIdx.y * X;
    Cize j = Index1;
    if (X <= j) return;

    Cize value = Y;
    for (Cize i = Y - 1; i >= 0; --i)
        if (input[i * X + j] > thresh) value = i;
    output[j] = value;
}

template <typename I, typename O>
void rank_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
    rank_gather<<<Config1(1, input.z, input.x)>>>(input.y, input.x, input.data, output.data, threshold);
}

template <typename I, typename O>
__global__ void rank_scatter(Cize Y, Cize X, PTR(I, input), PTR(O, output))
{
    input += blockIdx.y * X, output += blockIdx.y * Y * X;
    Cize j = Index1;
    if (X <= j) return;

    Cize value = input[j];
    for (Cize i = 0; i < Y; ++i) output[i * X + j] = (i >= value);
}

template <typename I, typename O>
void rank_scatter(Vec2<I> input, Vec3<O> output)
{
    rank_scatter<<<Config1(1, output.z, output.x)>>>(output.y, output.x, input.data, output.data);
}

template <typename I, typename O>
__global__ void rate_gather(Cize Y, Cize X, PTR(I, input), PTR(O, output), I threshold)
{
    input += (blockIdx.y * Y + Y - 1) * X, output += blockIdx.y * X;
    Cize idx = Index1D(O), end = min(X, idx + Block1D(O));
    for (Cize i = idx; i < end; i += Thread1D) output[i] = input[i];
}

template <typename I, typename O>
void rate_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
    rate_gather<<<Config1D(O, 1, input.z, input.x)>>>(input.y, input.x, input.data, output.data, threshold);
}
}  // namespace CUDA

void cuda_rank_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rank_gather<I Comma O>(input, output, threshold)));
}
void cuda_rank_scatter(Dyn2 input, Dyn3 output)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rank_scatter<I Comma O>(input, output)));
}
void cuda_rate_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rate_gather<I Comma O>(input, output, threshold)));
}
}  // namespace Core
}  // namespace Spyker
