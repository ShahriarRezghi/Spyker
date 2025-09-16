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

namespace Spyker
{
namespace Core
{
namespace CPU
{
template <typename I, typename O>
void rank_gather(ARG2(I, input), ARG1(O, output), I threshold)
{
    VEC2(I, input) VEC1(O, output);

    {
        I *id = input(input.y - 1, 0);
        auto init = O(input.y), time = O(input.y - 1);
        for (Size j = 0; j < input.x; ++j) output(j) = (id[j] > threshold) ? time : init;
    }

    for (Size i = input.y - 2; i >= 0; --i)
    {
        auto time = O(i);
        I *id = input(i, 0);
        for (Size j = 0; j < input.x; ++j)
            if (id[j] > threshold) output(j) = time;
    }
}

template <typename I, typename O>
void rank_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i) rank_gather(ARG(input(i)), ARG(output(i)), threshold);
}

template <typename I, typename O>
void rank_scatter(ARG1(I, input), ARG2(O, output))
{
    VEC1(I, input) VEC2(O, output);

    for (Size i = 0; i < output.y; ++i)
    {
        U32 time = i;
        O *od = output(i, 0);
        for (Size j = 0; j < output.x; ++j) od[j] = (time >= input(j));
    }
}

template <typename I, typename O>
void rank_scatter(Vec2<I> input, Vec3<O> output)
{
#pragma omp parallel for
    for (Size i = 0; i < output.z; ++i) rank_scatter(ARG(input(i)), ARG(output(i)));
}

template <typename I, typename O>
void rate_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i) copy(input(i, input.y - 1), output(i));
}
}  // namespace CPU

void cpu_rank_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rank_gather<I Comma O>(input, output, threshold)));
}
void cpu_rank_scatter(Dyn2 input, Dyn3 output)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rank_scatter<I Comma O>(input, output)));
}
void cpu_rate_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rate_gather<I Comma O>(input, output, threshold)));
}
}  // namespace Core
}  // namespace Spyker
