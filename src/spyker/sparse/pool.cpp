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
