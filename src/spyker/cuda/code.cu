// BSD 3-Clause License
//
// Copyright (c) 2022-2025, Shahriar Rezghi <shahriar.rezghi.sh@gmail.com>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "base.cuh"
//
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace Spyker
{
namespace Core
{
namespace CUDA
{
template <typename T>
__global__ void latency(Cize Y, Cize X, PTR(T, input), PTR(U16, output), PTR(T, minval), PTR(T, maxval))
{
    input += blockIdx.y * X, output += blockIdx.y * X;
    typename ToFloat<T>::Type final = Y + Epsilon, scale = final / (maxval[blockIdx.y] - minval[blockIdx.y]);
    Cize idx = Index1D(T), end = min(X, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) output[i] = final - (input[i] - minval[blockIdx.y]) * scale;
}

template <typename T>
__global__ void latency(Cize Y, Cize X, PTR(U16, input), PTR(T, output))
{
    input += blockIdx.z * X, output += (blockIdx.z * Y + blockIdx.y) * X;
    Cize idx = Index1D(T), end = min(X, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) output[i] = (input[i] <= blockIdx.y);
}

template <typename I, typename O>
void latency(Vec2<I> input_, Vec3<O> output)
{
    auto input = init<U16>(input_.y, input_.x);
    auto minmax = init<I>(2, input.y, maxsize<I>(input.x));
    auto min = minval(input_, minmax(0).data);
    auto max = maxval(input_, minmax(1).data);

    latency<<<Config1D(I, 1, input.y, input.x)>>>(output.y, output.x, input_.data, input.data, min.data, max.data);
    latency<<<Config1D(I, output.z, output.y, output.x)>>>(output.y, output.x, input.data, output.data);
    deinit(input, minmax);
}

template <typename T>
struct Comp
{
    T *data;
    Comp(T *data) : data(data) {}
    inline __device__ bool operator()(Cize A, Cize B) { return data[A] > data[B]; }
};

__global__ void sorted(Cize size, PTR(U32, index))
{
    index += blockIdx.y * size;
    Cize idx = Index1D(U32), end = min(size, idx + Block1D(U32));
    for (Cize i = idx; i < end; i += Thread1D) index[i] = i;
}

__global__ void sorted(Cize size, PTR(U32, xd), PTR(U16, od), PTR(F32, scale))
{
    xd += blockIdx.y * size, od += blockIdx.y * size;
    Cize idx = Index1D(F32), end = min(size, idx + Block1D(F32));
    for (Cize i = idx; i < end; i += Thread1D) od[xd[i]] = i * scale[blockIdx.y];
}

template <typename T>
void sorted(Vec2<T> input, Vec2<U32> index, Vec2<U16> output, Vec1<F32> scale, Size time)
{
    std::vector<F32> scale_(scale.x);
    sorted<<<Config1D(U32, 1, input.y, input.x)>>>(input.x, index.data);
    for (Size i = 0; i < input.y; ++i)
    {
        thrust::device_ptr<U32> ptr(index(i).data);
        thrust::sort(ptr, ptr + input.x, Comp<T>(input(i).data));

        auto min = d2h(index(i, index.x - 1));
        Size idx = thrust::lower_bound(ptr, ptr + input.x, min, Comp<T>(input(i).data)) - ptr;
        scale_[i] = F32(time + Epsilon) / idx;
    }

    cpu2cuda(scale.size() * sizeof(F32), scale_.data(), scale.data);
    sorted<<<Config1D(F32, 1, input.y, input.x)>>>(input.x, index.data, output.data, scale.data);
}

template <typename I, typename O>
void sorted(Vec2<I> input_, Vec3<O> output)
{
    auto input = init<U16>(input_.y, input_.x);
    auto index = init<U32>(input.y, input.x);
    auto scale = init<F32>(input.y);

    sorted(input_, index, input, scale, output.y);
    latency<<<Config1D(I, output.z, output.y, output.x)>>>(output.y, output.x, input.data, output.data);
    deinit(input, index, scale);
}
}  // namespace CUDA

void cuda_rank_code(Dyn2 input, Dyn3 output, bool sort)
{
    if (sort)
    {
        IfType(I, input.type, IfType(O, output.type, CUDA::sorted<I Comma O>(input, output)));
    }
    else
    {
        IfType(I, input.type, IfType(O, output.type, CUDA::latency<I Comma O>(input, output)));
    }
}

#define BINS 1024

namespace CUDA
{
using State = curandState;

__global__ void piosson_curand(Cize len, PTR(State, state), Size time)
{
    Cize i = Index1;
    if (i < len) curand_init(time, i, 0, state + i);
}

struct Poisson
{
    Vec2<U16> data;
    Vec2<State> rand;

    Poisson(Size time, Len2 dim)
    {
        auto old = poisson_create(time, BINS);
        data = init<U16>(old.y, old.x);
        cpu2cuda(data.size() * sizeof(U16), old.data, data.data);
        cpu_dealloc(old.data);

        rand = init<State>(dim.y, dim.x);
        piosson_curand<<<Config1(1, 1, rand.size())>>>(rand.size(), rand.data, Generator());
    }
    ~Poisson() { deinit(data, rand); }

    bool comp(Cize time, Len2 dim) { return data.y == time && rand.len() == dim; }
};

std::vector<std::shared_ptr<Poisson>> pois_handle;

void poisson_clear() { pois_handle.clear(); }

Poisson &pois_find(Size time, Len2 dim)
{
    for (auto pois : pois_handle)
        if (pois->comp(time, dim)) return *pois.get();
    pois_handle.push_back(std::shared_ptr<Poisson>(new Poisson(time, dim)));
    return *pois_handle.back().get();
}

template <typename T>
__global__ void poisson(Cize Y, Cize X, PTR(U16, input), PTR(T, output), PTR(State, rand), PTR(U16, data))
{
    input += blockIdx.y * X, rand += blockIdx.y * X, output += blockIdx.y * Y * X;
    Cize j = Index1;
    if (X <= j) return;

    State state = rand[j];
    U16 value = 0, *dd = data + input[j] * BINS;
    U16 time = dd[curand(&state) & (BINS - 1)];
    for (Cize i = 0; i < Y; ++i)
    {
        if (time == i) value += 1, time += dd[curand(&state) & (BINS - 1)];
        output[i * X + j] = value;
    }
    rand[j] = state;
}

template <typename I, typename O>
void poisson(Vec2<I> input, Vec3<O> output)
{
    Poisson &pois = pois_find(output.y, input.len());
    auto temp = init<U16>(input.y, input.x);
    auto minmax = init<I>(2, input.y, maxsize<I>(input.x));
    auto min = minval(input, minmax(0).data);
    auto max = maxval(input, minmax(1).data);

    latency<<<Config1D(I, 1, input.y, input.x)>>>(output.y, output.x, input.data, temp.data, min.data, max.data);
    poisson<<<Config1(1, input.y, input.x)>>>(output.y, output.x, temp.data, output.data, pois.rand.data,
                                              pois.data.data);

    deinit(input, minmax);
}

template <typename I, typename O>
void poissort(Vec2<I> input, Vec3<O> output)
{
    Poisson &pois = pois_find(output.y, input.len());
    auto temp = init<U16>(input.y, input.x);
    auto index = init<U32>(input.y, input.x);
    auto scale = init<F32>(input.y);

    sorted(input, index, temp, scale, output.y);
    poisson<<<Config1(1, input.y, input.x)>>>(output.y, output.x, temp.data, output.data, pois.rand.data,
                                              pois.data.data);
    deinit(temp, index, scale);
}
}  // namespace CUDA

void cuda_rate_code(Dyn2 input, Dyn3 output, bool sort)
{
    if (sort)
    {
        IfType(I, input.type, IfType(O, output.type, CUDA::poissort<I Comma O>(input, output)));
    }
    else
    {
        IfType(I, input.type, IfType(O, output.type, CUDA::poisson<I Comma O>(input, output)));
    }
}
void cuda_poisson_clear() { CUDA::poisson_clear(); }
}  // namespace Core
}  // namespace Spyker
