#include "base.h"
//
#include <algorithm>
#include <limits>

namespace Spyker
{
namespace Core
{
namespace CPU
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

template <typename T>
void latency(ARG1(U16, temp), ARG2(T, output))
{
    VEC1(U16, temp) VEC2(T, output);

    for (Size i = 0; i < output.y; ++i)
    {
        U16 max = i;
        T *od = output(i, 0);
        for (Size j = 0; j < output.x; ++j) od[j] = (temp(j) <= max);
    }
}

template <typename I, typename O>
void latency(Vec2<I> input, Vec3<O> output)
{
    BatchSize(input.y);
    auto temp = init<U16>(batch_size, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.y; ++i)
    {
        Size batch_index = BatchIndex(i);
        latency(ARG(input(i)), ARG(temp(batch_index)), output.y);
        latency(ARG(temp(batch_index)), ARG(output(i)));
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

template <typename I, typename O>
void sorted(Vec2<I> input, Vec3<O> output)
{
    BatchSize(input.y);
    auto temp = init<U16>(batch_size, input.x);
    auto index = init<U32>(batch_size, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.y; ++i)
    {
        Size batch_index = BatchIndex(i);
        sorted(ARG(input(i)), ARG(index(batch_index)), ARG(temp(batch_index)), output.y);
        latency(ARG(temp(batch_index)), ARG(output(i)));
    }

    deinit(temp, index);
}
}  // namespace CPU

void cpu_rank_code(Dyn2 input, Dyn3 output, bool sort)
{
    if (sort)
    {
        IfType(I, input.type, IfType(O, output.type, CPU::sorted<I Comma O>(input, output)));
    }
    else
    {
        IfType(I, input.type, IfType(O, output.type, CPU::latency<I Comma O>(input, output)));
    }
}

// Must be a multiple of two
#define BINS 1024

namespace CPU
{
using Random = std::mt19937;

struct Poisson
{
    Vec2<U16> data;

    Poisson(Size time)  //
    {
        data = poisson_create(time, BINS);
    }
    ~Poisson()
    {
        if (data.data != nullptr) deinit(data);
    }
    bool comp(Size time) { return data.y == time; }
};

std::vector<std::shared_ptr<Poisson>> pois_handle;

void poisson_clear() { pois_handle.clear(); }

Poisson &pois_find(Size time)
{
    for (auto pois : pois_handle)
        if (pois->comp(time)) return *pois.get();
    pois_handle.push_back(std::shared_ptr<Poisson>(new Poisson(time)));
    return *pois_handle.back().get();
}

inline U16 draw(Size i, Random &random, Poisson &pois)
{
    auto index = random() & (BINS - 1);
    return *pois.data(i, index);
}

template <typename T>
void poisson(ARG1(U16, input), ARG1(U16, times), ARG1(U16, values), ARG2(T, output), Random &random, Poisson &pois)
{
    VEC1(U16, input) VEC1(U16, times) VEC1(U16, values) VEC2(T, output);

    for (Size i = 0; i < output.y; ++i)
    {
        for (Size j = 0; j < output.x; ++j)
            if (times(j) == i) values(j) += 1, times(j) += draw(input(j), random, pois);

        copy(values, output(i));
    }
}

template <typename I, typename O>
void poisson(Vec2<I> input, Vec3<O> output)
{
    auto rnd = Generator();
    Poisson &pois = pois_find(output.y);
    auto temp = init<U16>(input.y, input.x);
    auto times = init<U16>(input.y, input.x);
    auto values = init<U16>(input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.y; ++i)
    {
        Random random(rnd + i);
        fill(values(i), U16(0));
        copy(input(i), times(i));
        latency(ARG(input(i)), ARG(temp(i)), output.y);
        poisson(ARG(temp(i)), ARG(times(i)), ARG(values(i)), ARG(output(i)), random, pois);
    }
    deinit(temp, times, values);
}

template <typename I, typename O>
void poissort(Vec2<I> input, Vec3<O> output)
{
    auto rnd = Generator();
    Poisson &pois = pois_find(output.y);
    auto temp = init<U16>(input.y, input.x);
    auto index = init<U32>(input.y, input.x);
    auto times = init<U16>(input.y, input.x);
    auto values = init<U16>(input.y, input.x);

#pragma omp parallel for
    for (Size i = 0; i < input.y; ++i)
    {
        Random random(rnd + i);
        fill(values(i), U16(0));
        copy(input(i), times(i));
        sorted(ARG(input(i)), ARG(index(i)), ARG(temp(i)), output.y);
        poisson(ARG(temp(i)), ARG(times(i)), ARG(values(i)), ARG(output(i)), random, pois);
    }
    deinit(temp, index, times, values);
}
}  // namespace CPU

void cpu_rate_code(Dyn2 input, Dyn3 output, bool sort)
{
    if (sort)
    {
        IfType(I, input.type, IfType(O, output.type, CPU::poissort<I Comma O>(input, output)));
    }
    else
    {
        IfType(I, input.type, IfType(O, output.type, CPU::poisson<I Comma O>(input, output)));
    }
}
void cpu_poisson_clear() { CPU::poisson_clear(); }
}  // namespace Core
}  // namespace Spyker
