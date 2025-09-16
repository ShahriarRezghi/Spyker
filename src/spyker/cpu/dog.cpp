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
template <typename T>
void dog(ARG1(T, input1), ARG1(T, input2), ARG1(T, output))
{
    VEC1(T, input1) VEC1(T, input2) VEC1(T, output);
    for (Size i = 0; i < output.x; ++i) output(i) = std::max<T>(input1(i) - input2(i), T(0));
}

template <typename T>
void dog(Vec4<T> input, Vec4<T> kernel, Vec4<T> output_, Len4 pad)
{
    auto middle_ = init<T>(output_.t, kernel.t, output_.y, output_.x);
    cpu_conv(todyn(input), todyn(kernel), todyn(middle_), {1, 1}, pad);
    Vec3<T> middle(middle_.data, middle_.t, 2, middle_.z / 2 * middle_.y * middle_.x);
    Vec2<T> output(output_.data, output_.t, output_.z * output_.y * output_.x);

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
        input1(i) = std::max<T>(diff, T(0));
        input2(i) = std::max<T>(-diff, T(0));
    }
}

template <typename T>
void log(Vec4<T> input, Vec4<T> kernel, Vec4<T> output_, Len4 pad)
{
    cpu_conv(todyn(input), todyn(kernel), todyn(output_), {1, 1}, pad);
    Vec3<T> output(output_.data, output_.t, 2, output_.z / 2 * output_.y * output_.x);

#pragma omp parallel for
    for (Size i = 0; i < output.z; ++i) log(ARG(output(i, 0)), ARG(output(i, 1)));
}
}  // namespace CPU

void cpu_dog(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad)
{
    IfType(T, kernel.type, CPU::dog<T>(input, kernel, output, pad));
}
void cpu_log(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad)
{
    IfType(T, kernel.type, CPU::log<T>(input, kernel, output, pad));
}
void cpu_gabor(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad) { cpu_conv(input, kernel, output, {1, 1}, pad); }
}  // namespace Core
}  // namespace Spyker
