#include "base.h"
//

namespace Spyker
{
namespace Core
{
namespace CPU
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

template <typename I, typename O>
void rank_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i) rank_gather(ARG(input(i)), ARG(output(i)), threshold);
}

template <typename I, typename O>
void rank_scatter(ARG1(I, input), ARG2(O, output))
{
    VEC1(I, input) VEC2(O, output);

    for (Size i = 0; i < output.y; ++i)
    {
        U32 time = i;
        O *od = output(i, 0);
        for (Size j = 0; j < output.x; ++j) od[j] = (time >= input(j));
    }
}

template <typename I, typename O>
void rank_scatter(Vec2<I> input, Vec3<O> output)
{
#pragma omp parallel for
    for (Size i = 0; i < output.z; ++i) rank_scatter(ARG(input(i)), ARG(output(i)));
}

template <typename I, typename O>
void rate_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
#pragma omp parallel for
    for (Size i = 0; i < input.z; ++i) copy(input(i, input.y - 1), output(i));
}
}  // namespace CPU

void cpu_rank_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rank_gather<I Comma O>(input, output, threshold)));
}
void cpu_rank_scatter(Dyn2 input, Dyn3 output)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rank_scatter<I Comma O>(input, output)));
}
void cpu_rate_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CPU::rate_gather<I Comma O>(input, output, threshold)));
}
}  // namespace Core
}  // namespace Spyker
