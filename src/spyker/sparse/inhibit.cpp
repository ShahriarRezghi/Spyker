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

namespace Spyker
{
namespace Core
{
void sparse_inhibit(Sparse5 input, Sparse5 output, Vec2<U16> channel, Size i)
{
    for (Size j = 0; j < input.t; ++j)
        for (Spridx index : input(i, j))
            if (*channel(index.y, index.x) == U16(-1)) *channel(index.y, index.x) = index.z;

    for (Size j = 0; j < input.t; ++j)
    {
        Sparse3 &od = output(i, j);
        for (Spridx index : input(i, j))
            if (*channel(index.y, index.x) == index.z) od.add(index);
    }
}

void sparse_inhibit(Sparse5 input, Sparse5 output)
{
    BatchSize(input.u);
    auto channel = CPU::init<U16>(batch_size, input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
    {
        Size batch_index = BatchIndex(i);
        CPU::fill(channel(batch_index), U16(-1));
        sparse_inhibit(input, output, channel(batch_index), i);
    }

    CPU::deinit(channel);
}
}  // namespace Core
}  // namespace Spyker
