#include "base.h"

namespace Spyker
{
namespace Core
{
namespace Sparse
{
struct Block
{
    U16 z, y, x;
    U16 ys, ye, xs, xe;
};

Size conv(Size len, PTR(Sparse3, input), PTR(Size, sizes))
{
    sizes[0] = 0;
    for (Size i = 0; i < len - 1; ++i)  //
        sizes[i + 1] = sizes[i] + input[i].size;
    return sizes[len - 1] + input[len - 1].size;
}

void conv(Sparse5 input, Vec1<Size> sizes, Vec1<Block> blocks, Len2 output, Len2 kernel, Len4 pad)
{
#pragma omp parallel for
    for (Size i = 0; i < sizes.x; ++i)
    {
        Spridx *id = input.data[i].data;
        Block *bd = blocks.data + sizes(i);
        for (Size j = 0; j < input.data[i].size; ++j)
        {
            Spridx index = id[j];
            Block &block = bd[j];
            block.z = index.z, block.y = index.y + pad.t, block.x = index.x + pad.z;
            block.ys = std::max(Size(block.y) - kernel.y + 1, Size(0));
            block.ye = std::min(output.y, Size(block.y + 1));
            block.xs = std::max(Size(block.x) - kernel.x + 1, Size(0));
            block.xe = std::min(output.x, Size(block.x + 1));
        }
    }
}

void conv(Sparse5 input, Vec1<Size> sizes, Vec1<Block> blocks, Vec3<F32> kernel, F32 threshold, Sparse5 output,
          Vec2<F32> temp, Size i, Size c)
{
    fill(temp, F32(0));
    for (Size j = 0; j < input.t; ++j)
    {
        Block *bd = blocks.data + sizes(i * input.t + j);
        for (Size k = 0; k < input(i, j).size; ++k)
        {
            Block block = bd[k];
            auto kd = kernel(block.z);
            for (Size k = block.ys; k < block.ye; ++k)
            {
                F32 *td = temp(k, 0);
                for (Size t = block.xs; t < block.xe; ++t)  //
                    td[t] += *kd(block.y - k, block.x - t);
            }
        }

        Sparse3 &od = output(i, j);
        if (input(i, j).size > 0)
            for (Size k = 0; k < temp.y; ++k)
            {
                F32 *td = temp(k, 0);
                for (Size t = 0; t < temp.x; ++t)
                    if (td[t] > threshold)
                    {
                        od.add(Spridx(c, k, t));
                        td[t] = Limits<F32>::min();
                    }
            }
    }
}

void conv(Sparse5 input, Vec4<F32> kernel, F32 threshold, Sparse5 output, Len2 stride, Len4 pad)
{
    SpykerCompare(stride.y, ==, 1, "Core::Conv", "Stride must be 1 in sparse conv.");
    SpykerCompare(stride.x, ==, 1, "Core::Conv", "Stride must be 1 in sparse conv.");

    BatchSize(output.u);
    auto sizes = init<Size>(input.u * input.t);
    auto temp = init<F32>(batch_size, output.y, output.x);
    auto blocks = init<Block>(conv(sizes.x, input.data, sizes.data));
    conv(input, sizes, blocks, {output.y, output.x}, {kernel.y, kernel.x}, pad);

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
        for (Size j = 0; j < kernel.t; ++j)  //
            conv(input, sizes, blocks, kernel(j), threshold, output, temp(BatchIndex(i)), i, j);

    deinit(sizes, blocks, temp);
}
}  // namespace Sparse

void sparse_conv(Sparse5 input, Dyn4 kernel, Scalar threshold, Sparse5 output, Len2 stride, Len4 pad)
{
    Sparse::conv(input, kernel, threshold, output, stride, pad);
}
}  // namespace Core
}  // namespace Spyker
