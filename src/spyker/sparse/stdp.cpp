#include "base.h"

namespace Spyker
{
namespace Core
{
using Configs = std::vector<STDPConfig>;

namespace Sparse
{
struct Block
{
    Size c, t, z;
    Size ys, ye, xs, xe;
};

void stdp(Vec1<Block> blocks, const Winners &winners, Vec1<Size> sizes, Len2 kernel)
{
    Size index = 0;
    for (const auto &list : winners)
        for (const auto &winner : list)
        {
            Block &block = blocks(index++);
            block.c = winner.c, block.t = winner.t, block.z = winner.z;
            block.ys = winner.y, block.ye = winner.y + kernel.y - 1;
            block.xs = winner.x, block.xe = winner.x + kernel.x - 1;
        }

    sizes(0) = 0;
    for (Size i = 0; i < sizes.x - 1; ++i)  //
        sizes(i + 1) = sizes(i) + winners[i].size();
}

void stdp(Vec1<U16> direct, Size time)
{
    for (Size i = 0; i < direct.x; ++i) direct(i) = (direct(i) <= time);
}

void stdp(Sparse5 input, Vec1<Block> blocks, Vec4<U16> direct, Len4 pad, Size i)
{
    fill(direct, U16(input.t));
    for (Size j = 0; j < input.t; ++j)
        for (Spridx index : input(i, j))
        {
            index.y += pad.t, index.x += pad.z;
            for (Size k = 0; k < blocks.x; ++k)
            {
                Block block = blocks.data[k];
                bool Y = block.ys <= index.y && index.y <= block.ye;
                bool X = block.xs <= index.x && index.x <= block.xe;
                if (Y && X) *direct(k, index.z, index.y - block.ys, index.x - block.xs) = j;
            }
        }

    for (Size j = 0; j < blocks.x; ++j)  //
        stdp(Vec1<U16>(direct(j).data, direct(j).size()), blocks(j).t);
}

void stdp(ARG1(U16, direct), ARG1(F32, kernel), STDPConfig config)
{
    VEC1(U16, direct) VEC1(F32, kernel);

    for (Size i = 0; i < direct.x; ++i)
    {
        F32 mult = (config.stabilize ? (kernel(i) - config.lower) * (config.upper - kernel(i)) : 1);
        kernel(i) += (direct(i) ? config.positive : config.negative) * mult;
        kernel(i) = std::max(config.lower, std::min(F64(kernel(i)), config.upper));
    }
}

void stdp(Vec4<F32> kernel, Vec1<Block> blocks, Vec4<U16> direct, const Configs &configs)
{
    for (Size i = 0; i < blocks.x; ++i)
    {
        Block block = blocks(i);
        if (block.c < 0) continue;
        SpykerCompare(block.c, <, configs.size(), "Sparse::STDP", "STDP config index is out of range.");
        stdp(ARG(Vec1<U16>(direct(i).data, direct(i).size())),  //
             ARG(Vec1<F32>(kernel(block.z).data, kernel(block.z).size())), configs[block.c]);
    }
}

void stdp(Sparse5 input, Vec4<F32> kernel, const Configs &configs, const Winners &winners, Len4 pad)
{
    Size total = 0;
    for (Size i = 0; i < input.u; ++i) total += winners[i].size();

    auto blocks = init<Block>(total);
    auto sizes = init<Size>(input.u);
    auto direct = init<U16>(total, kernel.z, kernel.y, kernel.x);
    stdp(blocks, winners, sizes, {kernel.y, kernel.x});

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
    {
        Size size = winners[i].size();
        Vec1<Block> blocks_ = {blocks.data + sizes(i), size};
        Vec4<U16> direct_ = {direct(sizes(i)).data, size, direct.z, direct.y, direct.x};
        stdp(input, blocks_, direct_, pad, i);
    }
    stdp(kernel, blocks, direct, configs);
    deinit(blocks, sizes, direct);
}
}  // namespace Sparse

void sparse_stdp(Sparse5 input, Dyn4 kernel, const Configs &configs, const Winners &winners, Len4 pad)
{
    Sparse::stdp(input, kernel, configs, winners, pad);
}
}  // namespace Core
}  // namespace Spyker
