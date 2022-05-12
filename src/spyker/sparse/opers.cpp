#include "base.h"

namespace Spyker
{
namespace Core
{
namespace Sparse
{
template <typename I, typename O>
void rank_gather(ARG2(I, input), ARG1(O, output), I threshold)
{
    VEC2(I, input) VEC1(O, output);

    {
        I *id = input(input.y - 1, 0);
        auto init = O(input.y), time = O(input.y - 1);
        for (Size j = 0; j < input.x; ++j) output(j) = (id[j] > threshold) ? time : init;
    }

    for (Size i = input.y - 2; i >= 0; --i)
    {
        auto time = O(i);
        I *id = input(i, 0);
        for (Size j = 0; j < input.x; ++j)
            if (id[j] > threshold) output(j) = time;
    }
}

template <typename T>
void convert(Vec3<T> input, Sparse5 output, Size i)
{
    for (Size j = 0; j < input.z; ++j)
        for (Size k = 0; k < input.y; ++k)
        {
            auto id = input(j, k);
            for (Size t = 0; t < input.x; ++t)
                if (id(t) < output.t) output(i, id(t)).add(Spridx(j, k, t));
        }
}

template <typename T>
void convert(Vec5<T> input, Sparse5 output, T threshold)
{
    BatchSize(input.u);
    auto temp = CPU::init<U16>(batch_size, input.z, input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
    {
        auto batch_index = BatchIndex(i);
        Len2 dim = {output.t, output.z * output.y * output.x};
        rank_gather(ARG(Vec2<T>(input(i).data, dim.y, dim.x)),  //
                    ARG(Vec1<U16>(temp(batch_index).data, dim.x)), threshold);
        convert(temp(batch_index), output, i);
    }

    deinit(temp);
}

template <typename T>
void convert(Sparse5 input, Vec4<T> output, Size i)
{
    for (Size j = 0; j < input.t; ++j)
    {
        auto od = output(j);
        if (j == 0) fill(output(j), T(0));
        if (j != 0) copy(output(j - 1), output(j));
        for (Spridx index : input(i, j)) *od(index.z, index.y, index.x) = T(1);
    }
}

template <typename T>
void convert(Sparse5 input, Vec5<T> output)
{
#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i) convert(input, output(i), i);
}

template <typename T>
void gather(Sparse5 input, Vec4<T> output)
{
#pragma omp parallel for
    for (Size i = 0; i < input.u; ++i)
    {
        auto od = output(i);
        fill(od, T(input.t));
        for (Size j = input.t - 1; j >= 0; --j)
            for (Spridx index : input(i, j)) *od(index.z, index.y, index.x) = j;
    }
}
}  // namespace Sparse

void sparse_convert(Dyn5 input, Sparse5 output, Scalar threshold)  //
{
    IfType(T, input.type, Sparse::convert<T>(input, output, threshold));
}
void sparse_convert(Sparse5 input, Dyn5 output)  //
{
    IfType(T, output.type, Sparse::convert<T>(input, output));
}
void sparse_gather(Sparse5 input, Dyn4 output)  //
{
    IfType(T, output.type, Sparse::gather<T>(input, output));
}
}  // namespace Core
}  // namespace Spyker
