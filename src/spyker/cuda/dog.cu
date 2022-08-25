#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
template <typename T>
__global__ void dog(Cize size, PTR(T, input), PTR(T, output))
{
    input += blockIdx.y * 2 * size, output += blockIdx.y * size;
    Cize i = Index1;
    if (size <= i) return;
    output[i] = cmax(input[i] - input[i + size], T(0));
}

template <typename T>
void dog(Vec4<T> input, Vec4<T> kernel, Vec4<T> output, Len4 pad)
{
    auto middle = init<T>(output.t, kernel.t, output.y, output.x);
    cuda_conv(todyn(input), todyn(kernel), todyn(middle), {1, 1}, pad);
    Len2 dim = {output.t, output.z * output.y * output.x};
    dog<<<Config1(1, dim.y, dim.x)>>>(dim.x, middle.data, output.data);
    deinit(middle);
}

template <typename T>
__global__ void log(Cize size, PTR(T, input))
{
    T *input1 = input + blockIdx.y * 2 * size, *input2 = input1 + size;
    Cize i = Index1;
    if (size <= i) return;

    T diff = input1[i] - input2[i];
    input1[i] = cmax(diff, T(0));
    input2[i] = cmax(-diff, T(0));
}

template <typename T>
void log(Vec4<T> input, Vec4<T> kernel, Vec4<T> output, Len4 pad)
{
    cuda_conv(todyn(input), todyn(kernel), todyn(output), {1, 1}, pad);
    Len2 dim = {output.t, output.z / 2 * output.y * output.x};
    log<<<Config1(1, dim.y, dim.x)>>>(dim.x, output.data);
}

template <typename T>
__global__ void zca_split(Cize size, PTR(T, input), PTR(T, output))
{
    input += blockIdx.y * size;
    T *od1 = output + blockIdx.y * 2 * size, *od2 = od1 + size;
    Cize i = Index1;
    if (size <= i) return;

    od1[i] = cmax(T(input[i]), T(0));
    od2[i] = cmax(T(-input[i]), T(0));
}

template <typename T>
void zca_split(Vec2<T> input, Vec3<T> output)
{
    zca_split<<<Config1(1, input.y, input.x)>>>(input.x, input.data, output.data);
}
}  // namespace CUDA

void cuda_dog(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad)
{
    IfType(T, kernel.type, CUDA::dog<T>(input, kernel, output, pad));
}
void cuda_log(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad)
{
    IfType(T, kernel.type, CUDA::log<T>(input, kernel, output, pad));
}
void cuda_gabor(Dyn4 input, Dyn4 kernel, Dyn4 output, Len4 pad) { cuda_conv(input, kernel, output, {1, 1}, pad); }

void cuda_zca_fit(Dyn2 input, Dyn1 mean, Dyn2 trans, Scalar epsilon, bool transform)
{
    SpykerAssert(false, "CUDA::ZCA", "ZCA functionality is not implemented on CUDA.");
}
void cuda_zca_trans(Dyn2 input, Dyn1 mean, Dyn2 trans)
{
    SpykerAssert(false, "CUDA::ZCA", "ZCA functionality is not implemented on CUDA.");
}
void cuda_zca_split(Dyn2 input, Dyn3 output) { IfType(T, input.type, CUDA::zca_split<T>(input, output)); }
}  // namespace Core
}  // namespace Spyker
