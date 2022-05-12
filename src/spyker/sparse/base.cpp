#include "base.h"

namespace Spyker
{
namespace Core
{
Size sparse_elemsize(Sparse5 sparse)
{
    Size size = 0;
    for (Size i = 0; i < sparse.u; ++i)
        for (Size j = 0; j < sparse.t; ++j) size += sparse(i, j).size;
    return size;
}

Size sparse_memsize(Sparse5 sparse)
{
    Size size = 0;
    for (Size i = 0; i < sparse.u; ++i)
        for (Size j = 0; j < sparse.t; ++j) size += sparse(i, j).max;
    return sizeof(Sparse5) + sparse.u * sparse.t * sizeof(Sparse3) + size * sizeof(Spridx);
}

void sparse_alloc(Sparse3 &input)
{
    input = Sparse3();
    input.data = (Spridx *)malloc(input.max * sizeof(Spridx));
}

Sparse5 sparse_alloc(Len5 len)
{
    Sparse5 output;
    output.u = len.u, output.t = len.t, output.z = len.z, output.y = len.y, output.x = len.x;
    Size size = len.u * len.t;
    output.data = (Sparse3 *)malloc(size * sizeof(Sparse3));
    for (Size i = 0; i < size; ++i) sparse_alloc(output.data[i]);
    return output;
}

void sparse_dealloc(Sparse5 sparse)
{
    Size size = sparse.u * sparse.t;
    for (Size i = 0; i < size; ++i) free(sparse.data[i].data);
    free(sparse.data);
}

void sparse_copy(Sparse5 input, Sparse5 output)
{
#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
        for (Size j = 0; j < input.t; ++j)
        {
            Sparse3 &id = input(i, j), &od = output(i, j);
            od.size = id.size, od.max = id.max;
            od.data = (Spridx *)realloc(od.data, od.max * sizeof(Spridx));
            std::copy(id.data, id.data + id.size, od.data);
        }
}
}  // namespace Core
}  // namespace Spyker
