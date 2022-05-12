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
