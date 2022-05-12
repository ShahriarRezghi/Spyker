#include "base.h"

namespace Spyker
{
namespace Core
{
namespace Sparse
{
template <typename T>
void latency(ARG1(T, input), ARG1(U16, temp), Size time)
{
    VEC1(T, input) VEC1(U16, temp);
    T min = input(0), max = input(0);

    for (Size i = 0; i < input.x; ++i)  //
        min = std::min(min, input(i)), max = std::max(max, input(i));

    F32 end = time + Epsilon, scale = end / (max - min);
    for (Size i = 0; i < input.x; ++i) temp(i) = end - scale * (input(i) - min);
}

void latency(Vec3<U16> input, Sparse5 output, Size i)
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
void latency(Vec4<T> input, Sparse5 output)
{
    BatchSize(input.t);
    auto temp = init<U16>(batch_size, input.z, input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.t; ++i)
    {
        Size batch_index = BatchIndex(i);
        latency(ARG(input(i).flat()), ARG(temp(batch_index).flat()), output.t);
        latency(temp(batch_index), output, i);
    }

    deinit(temp);
}

template <typename T>
Size sorted(ARG1(T, input), ARG1(U32, index))
{
    VEC1(T, input) VEC1(U32, index);
    auto comp = [&input](U32 i, U32 j) { return input(i) > input(j); };
    return std::distance(index.begin(), std::lower_bound(index.begin(), index.end(), index(index.x - 1), comp));
}

template <typename T>
void sorted(ARG1(T, input), ARG1(U32, index), ARG1(U16, temp), Size time)
{
    VEC1(T, input) VEC1(U32, index) VEC1(U16, temp);
    for (Size i = 0; i < index.x; ++i) index(i) = i;
    std::sort(index.begin(), index.end(), [&input](U32 i, U32 j) { return input(i) > input(j); });
    F32 scale = F32(time + Epsilon) / sorted(ARG(input), ARG(index));
    for (Size i = 0; i < temp.x; ++i) temp(index(i)) = i * scale;
}

template <typename T>
void sorted(Vec4<T> input, Sparse5 output)
{
    BatchSize(input.t);
    auto temp = init<U16>(batch_size, input.z, input.y, input.x);
    auto index = init<U32>(batch_size, input.z, input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.t; ++i)
    {
        Size batch_index = BatchIndex(i);
        Size size = input.z * input.y * input.x;
        sorted(ARG(input(i).flat()),            //
               ARG(index(batch_index).flat()),  //
               ARG(temp(batch_index).flat()), output.t);
        latency(temp(batch_index), output, i);
    }

    deinit(temp, index);
}
}  // namespace Sparse

void sparse_code(Dyn4 input, Sparse5 output, bool sort)
{
    if (sort)
    {
        IfType(T, input.type, Sparse::sorted<T>(input, output));
    }
    else
    {
        IfType(T, input.type, Sparse::latency<T>(input, output));
    }
}
}  // namespace Core
}  // namespace Spyker
