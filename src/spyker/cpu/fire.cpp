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
#include <omp.h>

namespace Spyker
{
namespace Core
{
namespace CPU
{
template <typename T>
void threshold(Size size, PTR(T, input), T threshold, T value)
{
    for (Size i = 0; i < size; ++i)
        if (input[i] <= threshold) input[i] = value;
}

template <typename T>
void threshold(Vec1<T> input, T threshold, T value)
{
    Size threads = omp_get_max_threads();
    Size blocks = (input.x + threads - 1) / threads;

#pragma omp parallel for
    for (Size i = 0; i < input.x; i += blocks)  //
        CPU::threshold(std::min(blocks, input.x - i), &input(i), threshold, value);
}

template <typename T>
void rank_infinite(Vec3<T> input, T value)
{
#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i)  //
        fill(Vec2<T>(input(i).data, input.y - 1, input.x), value);
}

template <typename I, typename O>
void rank_fire(Size size, PTR(I, input), PTR(O, output), I threshold)
{
    for (Size i = 0; i < size; ++i) output[i] = (input[i] > threshold);
}

template <typename I, typename O>
void rank_fire(Vec1<I> input, Vec1<O> output, I threshold)
{
    Size threads = omp_get_max_threads();
    Size blocks = (input.x + threads - 1) / threads;

#pragma omp parallel for
    for (Size i = 0; i < input.x; i += blocks)
        rank_fire(std::min(blocks, input.x - i), &input(i), &output(i), threshold);
}

template <typename I, typename O>
void rate_fire(ARG1(I, input), ARG1(I, values), ARG1(O, spikes), I threshold)
{
    VEC1(I, input) VEC1(I, values) VEC1(O, spikes);

    for (Size i = 0; i < input.x; ++i)
        if (values(i) + threshold < input(i))  //
            ++spikes(i), values(i) = input(i);
}

template <typename I, typename O>
void rate_fire(Vec3<I> input, Vec3<O> output, I threshold)
{
    BatchSize(input.z);
    auto values = init<I>(batch_size, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i)
    {
        Size batch_index = BatchIndex(i);
        auto spikes = output(i, output.y - 1);
        fill(values(batch_index), I(0)), fill(spikes, O(0));
        for (Size j = 0; j < input.y; ++j)
        {
            rate_fire(ARG(input(i, j)), ARG(values(batch_index)), ARG(spikes), threshold);
            if (j != output.y - 1) copy(spikes, output(i, j));
        }
    }

    deinit(values);
}

template <typename T>
void quantize(Size size, PTR(T, input), T lower, T middle, T upper)
{
    for (Size i = 0; i < size; ++i) input[i] = input[i] < middle ? lower : upper;
}

template <typename T>
void quantize(Vec1<T> input, T lower, T middle, T upper)
{
    Size threads = omp_get_max_threads();
    Size blocks = (input.x + threads - 1) / threads;

#pragma omp parallel for
    for (Size i = 0; i < input.x; i += blocks)  //
        quantize(std::min(blocks, input.x - i), &input(i), lower, middle, upper);
}
}  // namespace CPU

void cpu_threshold(Dyn1 input, Scalar threshold, Scalar value)
{
    IfType(T, input.type, CPU::threshold<T>(input, threshold, value));
}
void cpu_rank_infinite(Dyn3 input, Scalar value)  //
{
    IfType(T, input.type, CPU::rank_infinite<T>(input, value));
}
void cpu_rank_fire(Dyn1 input, Dyn1 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rank_fire<I Comma O>(input, output, threshold)));
}
void cpu_rate_fire(Dyn3 input, Dyn3 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rate_fire<I Comma O>(input, output, threshold)));
}
void cpu_quantize(Dyn1 input, Scalar lower, Scalar middle, Scalar upper)
{
    IfType(T, input.type, CPU::quantize<T>(input, lower, middle, upper));
}
}  // namespace Core
}  // namespace Spyker
