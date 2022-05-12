#include "base.h"

namespace Spyker
{
namespace Core
{
void sparse_pad(Sparse3 &input, Sparse3 &output, Len4 pad)
{
    output.size = input.size, output.max = input.max;
    output.data = (Spridx *)realloc(output.data, output.max * sizeof(Spridx));
    for (Size k = 0; k < input.size; ++k)
    {
        Spridx index = input(k);
        index.y += pad.t, index.x += pad.z;
        output(k) = index;
    }
}

void sparse_pad(Sparse5 input, Sparse5 output, Len4 pad)
{
#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
        for (Size j = 0; j < input.t; ++j)  //
            sparse_pad(input(i, j), output(i, j), pad);
}
}  // namespace Core
}  // namespace Spyker
