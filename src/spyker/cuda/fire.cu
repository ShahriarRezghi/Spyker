#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
template <typename T>
__global__ void threshold(Cize size, PTR(T, input), T threshold, T value)
{
    Cize idx = Index1D(T), end = min(size, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D)
        if (input[i] <= threshold) input[i] = value;
}

template <typename T>
void threshold(Vec1<T> input, T threshold, T value)
{
    CUDA::threshold<<<Config1D(T, 1, 1, input.x)>>>(input.x, input.data, threshold, value);
}

template <typename T>
__global__ void rank_infinite(Cize Y, Cize X, PTR(T, input), T value)
{
    input += (blockIdx.z * Y + blockIdx.y) * X;
    Cize idx = Index1D(T), end = min(X, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) input[i] = value;
}

template <typename T>
void rank_infinite(Vec3<T> input, T value)
{
    if (input.y <= 1) return;
    rank_infinite<<<Config1D(T, input.z, input.y - 1, input.x)>>>(input.y, input.x, input.data, value);
}

template <typename I, typename O>
__global__ void rank_fire(Cize size, PTR(I, input), PTR(O, output), I threshold)
{
    Cize idx = Index1D(I), end = min(size, idx + Block1D(I));
    for (Cize i = idx; i < end; i += Thread1D) output[i] = input[i] > threshold;
}

template <typename I, typename O>
void rank_fire(Vec1<I> input, Vec1<O> output, I threshold)
{
    rank_fire<<<Config1D(I, 1, 1, input.x)>>>(input.x, input.data, output.data, threshold);
}

template <typename I, typename O>
__global__ void rate_fire(Cize Y, Cize X, PTR(I, input), PTR(O, output), I threshold)
{
    input += blockIdx.y * Y * X, output += blockIdx.y * Y * X;
    Cize j = Index1;
    if (X <= j) return;

    O spike = 0;
    I value = 0;
    for (Cize i = 0; i < Y; ++i)
    {
        if (value + threshold < input[i * X + j])  //
            spike += 1, value = input[i * X + j];
        output[i * X + j] = spike;
    }
}

template <typename I, typename O>
void rate_fire(Vec3<I> input, Vec3<O> output, I threshold)
{
    rate_fire<<<Config1(1, input.z, input.x)>>>(input.y, input.x, input.data, output.data, threshold);
}

template <typename T>
__global__ void quantize(Cize size, PTR(T, input), T lower, T middle, T upper)
{
    Cize idx = Index1D(T), end = min(size, idx + Block1D(T));
    for (Cize i = idx; i < end; i += Thread1D) input[i] = input[i] < middle ? lower : upper;
}

template <typename T>
void quantize(Vec1<T> input, T lower, T middle, T upper)
{
    quantize<<<Config1D(T, 1, 1, input.x)>>>(input.x, input.data, lower, middle, upper);
}
}  // namespace CUDA

void cuda_threshold(Dyn1 input, Scalar threshold, Scalar value)
{
    IfType(T, input.type, CUDA::threshold<T>(input, threshold, value));
}
void cuda_rank_infinite(Dyn3 input, Scalar value)  //
{
    IfType(T, input.type, CUDA::rank_infinite<T>(input, value));
}
void cuda_rank_fire(Dyn1 input, Dyn1 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rank_fire<I Comma O>(input, output, threshold)));
}
void cuda_rate_fire(Dyn3 input, Dyn3 output, Scalar threshold)
{
    IfType(I, input.type, IfType(O, output.type, CUDA::rate_fire<I Comma O>(input, output, threshold)));
}
void cuda_quantize(Dyn1 input, Scalar lower, Scalar middle, Scalar upper)
{
    IfType(T, input.type, CUDA::quantize<T>(input, lower, middle, upper));
}
}  // namespace Core
}  // namespace Spyker
