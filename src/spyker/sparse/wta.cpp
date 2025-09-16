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
struct Block
{
    U16 z, ys, ye, xs, xe;
};

using Blocks = std::vector<Block>;

Block sparse_convwta(Spridx index, Len5 input, Len2 radius)
{
    Block block;
    block.z = index.z;
    block.ys = std::max(Size(0), index.y - radius.y);
    block.ye = std::min(input.y - 1, index.y + radius.y);
    block.xs = std::max(Size(0), index.x - radius.x);
    block.xe = std::min(input.x - 1, index.x + radius.x);
    return block;
}

bool sparse_convwta(const Blocks &blocks, Spridx index)
{
    for (const auto &block : blocks)
    {
        if (index.z == block.z) return false;
        if (index.y < block.ys || block.ye < index.y) continue;
        if (index.x < block.xs || block.xe < index.x) continue;
        return false;
    }
    return true;
}

std::vector<Winner> sparse_convwta(Sparse5 input, Len2 radius, Size count, Size i)
{
    Blocks blocks;
    Size found = 0;
    std::vector<Winner> winners;

    for (Size j = 0; j < input.t; ++j)
        for (Spridx index : input(i, j))
            if (sparse_convwta(blocks, index))
            {
                ++found;
                winners.push_back(Winner{0, j, index.z, index.y, index.x});
                if (found >= count) return winners;
                blocks.push_back(sparse_convwta(index, input.len(), radius));
            }

    return winners;
}

Winners sparse_convwta(Sparse5 input, Len2 radius, Size count)
{
    Winners winners(input.u);

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)  //
        winners[i] = sparse_convwta(input, radius, count, i);

    return winners;
}
}  // namespace Core
}  // namespace Spyker
