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
