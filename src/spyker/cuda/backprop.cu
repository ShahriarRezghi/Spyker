#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
// template <typename T>
//__global__ void backward(Cize X, PTR(T, input), PTR(T, output), PTR(T, minimum), PTR(I64, target), Cize time, T gamma)
//{
//     input += blockIdx.y * X, output += blockIdx.y * X;
//     Cize j = Index1;
//     if (X < j) return;

//    T minim = minimum[blockIdx.y], upper = minim + gamma, result = input[j];
//    if (min != time && result < upper) result = min(T(time), upper);
//    if (j == target) result = std::max(T(0), minim /*- gamma*/);
//    output[j] = (result - input[i]) / time;
//}

// template <typename T>
// void backward(Vec2<T> input, Vec2<T> output, Vec1<I64> target, Size time, T gamma)
//{
//     auto min = init<T>(input.y, maxsize<T>(input.x));
//     minval(input, min.data);
//     backward<<<Config1(1, input.y, input.x)>>>(  //
//         input.x, input.data, output.data, min.data, target.data, time, gamma);
//     deinit(max);
// }
}  // namespace CUDA

void cuda_backward(Dyn2 input, Dyn2 output, Dyn1 target, Size time, Scalar gamma)
{
    //    IfType(T, input.type, CUDA::backward<T>(input, output, target, time, gamma));
    SpykerAssert(false, "CUDA::Backprop", "Backpropagation is not implemented on CUDA.");
}
void cuda_labelize(Dyn3 input, Dyn1 output, Scalar threshold)
{
    SpykerAssert(false, "CUDA::Backprop", "Backpropagation is not implemented on CUDA.");
}
void cuda_fcbackward(Dyn2 kernel, Dyn2 input, Dyn2 output, Dyn2 grad, Dyn2 next, BPConfig &config)
{
    SpykerAssert(false, "CUDA::Backprop", "Backpropagation is not implemented on CUDA.");
}
void cuda_signfc(Dyn3 input, Dyn2 kernel, Dyn3 output)
{
    SpykerAssert(false, "CUDA::Backprop", "Backpropagation is not implemented on CUDA.");
}
}  // namespace Core
}  // namespace Spyker
