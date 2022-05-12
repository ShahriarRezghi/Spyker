#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
template <typename I, typename O>
__global__ void rank_gather(Cize Y, Cize X, PTR(I, input), PTR(O, output), I thresh)
{
    input += blockIdx.y * Y * X, output += blockIdx.y * X;
    Cize j = Index1;
    if (X <= j) return;

    Cize value = Y;
    for (Cize i = Y - 1; i >= 0; --i)
        if (input[i * X + j] > thresh) value = i;
    output[j] = value;
}

template <typename I, typename O>
void rank_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
    rank_gather<<<Config1(1, input.z, input.x)>>>(input.y, input.x, input.data, output.data, threshold);
}

template <typename I, typename O>
__global__ void rank_scatter(Cize Y, Cize X, PTR(I, input), PTR(O, output))
{
    input += blockIdx.y * X, output += blockIdx.y * Y * X;
    Cize j = Index1;
    if (X <= j) return;

    Cize value = input[j];
    for (Cize i = 0; i < Y; ++i) output[i * X + j] = (i >= value);
}

template <typename I, typename O>
void rank_scatter(Vec2<I> input, Vec3<O> output)
{
    rank_scatter<<<Config1(1, output.z, output.x)>>>(output.y, output.x, input.data, output.data);
}

template <typename I, typename O>
__global__ void rate_gather(Cize Y, Cize X, PTR(I, input), PTR(O, output), I threshold)
{
    input += (blockIdx.y * Y + Y - 1) * X, output += blockIdx.y * X;
    Cize idx = Index1D(O), end = min(X, idx + Block1D(O));
    for (Cize i = idx; i < end; i += Thread1D) output[i] = input[i];
}

template <typename I, typename O>
void rate_gather(Vec3<I> input, Vec2<O> output, I threshold)
{
    rate_gather<<<Config1D(O, 1, input.z, input.x)>>>(input.y, input.x, input.data, output.data, threshold);
}
}  // namespace CUDA

void cuda_rank_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rank_gather<I Comma O>(input, output, threshold)));
}
void cuda_rank_scatter(Dyn2 input, Dyn3 output)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rank_scatter<I Comma O>(input, output)));
}
void cuda_rate_gather(Dyn3 input, Dyn2 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rate_gather<I Comma O>(input, output, threshold)));
}
}  // namespace Core
}  // namespace Spyker
