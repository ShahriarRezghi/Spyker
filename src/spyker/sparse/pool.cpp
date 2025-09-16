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
void sparse_pool(Sparse5 input, Sparse5 output, Vec3<U8> check, Len2 kernel, Len4 pad, Size i)
{
    CPU::fill(check, U8(0));
    U16 limy = output.y, limx = output.x;
    for (Size j = 0; j < input.t; ++j)
        for (Spridx index : input(i, j))
        {
            index.y = (pad.t + index.y) / kernel.y;
            index.x = (pad.z + index.x) / kernel.x;
            if (index.y >= limy || index.x >= limx) continue;
            if (*check(index.z, index.y, index.x) == 1) continue;
            *check(index.z, index.y, index.x) = 1, output(i, j).add(index);
        }
}

void sparse_pool(Sparse5 input, Sparse5 output, Len2 kernel, Len2 stride, Len4 pad)
{
    SpykerCompare(stride.y, ==, kernel.y, "Core::Pool", "Stride must be equal to kernel in sparse pool.");
    SpykerCompare(stride.x, ==, kernel.x, "Core::Pool", "Stride must be equal to kernel in sparse pool.");

    BatchSize(output.u);
    auto check = CPU::init<U8>(batch_size, output.z, output.y, output.x);

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i) sparse_pool(input, output, check(BatchIndex(i)), kernel, pad, i);

    CPU::deinit(check);
}
}  // namespace Core
}  // namespace Spyker
