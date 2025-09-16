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
void rank_inhibit(ARG3(T, input), ARG2(T, maximum), ARG2(U16, channel))
{
    VEC3(T, input) VEC2(T, maximum) VEC2(U16, channel);

    for (Size i = 0; i < input.z; ++i) copy(input(i, 0), maximum(i));
    fill(channel, U16(0));

    for (Size i = 0; i < input.z; ++i)
        for (Size j = 1; j < input.y; ++j)
        {
            U16 ch = j;
            T *id = input(i, j, 0);
            T *md = maximum(i, 0);
            U16 *cd = channel(i, 0);

#pragma clang loop vectorize(enable) interleave(enable)
            for (Size k = 0; k < input.x; ++k)
                if (id[k] > md[k]) md[k] = id[k], cd[k] = ch;
        }
}

template <typename T>
void rank_inhibit(ARG2(T, maximum), ARG1(U16, sum), T threshold)
{
    VEC2(T, maximum) VEC1(U16, sum);

    fill(sum, U16(0));
    for (Size i = 0; i < maximum.y; ++i)
    {
        T *md = maximum(i, 0);
        for (Size j = 0; j < maximum.x; ++j) sum(j) += (md[j] > threshold);
    }
}

void rank_inhibit(ARG2(U16, channel), ARG1(U16, sum), ARG1(U16, index))
{
    VEC2(U16, channel) VEC1(U16, sum) VEC1(U16, index);

    for (Size i = 0; i < channel.x; ++i)
    {
        U16 time = channel.y - std::max(sum(i), U16(1));
        index(i) = *channel(time, i);
    }
}

template <typename T>
void rank_inhibit(ARG3(T, input), ARG1(U16, index))
{
    VEC3(T, input) VEC1(U16, index);

    for (Size i = 0; i < input.z; ++i)
        for (Size j = 0; j < input.y; ++j)
        {
            U16 channel = j;
            T *id = input(i, j, 0);
            for (Size k = 0; k < input.x; ++k) id[k] *= (index(k) == channel);
        }
}

template <typename T>
void rank_inhibit(Vec4<T> input, T threshold)
{
    BatchSize(input.t);
    auto maximum = init<T>(batch_size, input.z, input.x);
    auto channel = init<U16>(batch_size, input.z, input.x);
    auto sum = init<U16>(batch_size, input.x);
    auto index = init<U16>(batch_size, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.t; ++i)
    {
        Size bidx = BatchIndex(i);
        rank_inhibit(ARG(input(i)), ARG(maximum(bidx)), ARG(channel(bidx)));
        rank_inhibit(ARG(maximum(bidx)), ARG(sum(bidx)), threshold);
        rank_inhibit(ARG(channel(bidx)), ARG(sum(bidx)), ARG(index(bidx)));
        rank_inhibit(ARG(input(i)), ARG(index(bidx)));
    }

    deinit(maximum, channel, sum, index);
}
}  // namespace CPU

void cpu_rank_inhibit(Dyn4 input, Scalar threshold) { IfType(T, input.type, CPU::rank_inhibit<T>(input, threshold)); }
}  // namespace Core
}  // namespace Spyker
