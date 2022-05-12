#include "base.cuh"

namespace Spyker
{
namespace Core
{
namespace CUDA
{
void matmul_(F32 *A, F32 *B, F32 *C, bool AT, bool BT, int AR, int AC, int BR, int BC)
{
    if (!cublas_static) cublas_static = std::unique_ptr<cublas>(new cublas);

    F32 alpha = 1, beta = 0;
    int ATR = AT ? AC : AR, ATC = AT ? AR : AC, BTC = BT ? BR : BC;
    cublasOperation_t AT_ = AT ? CUBLAS_OP_T : CUBLAS_OP_N, BT_ = BT ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(cublas_static->handle, AT_, BT_, ATR, BTC, ATC, &alpha, A, AR, B, BR, &beta, C, ATR);
}

void matmul(F32 *A, F32 *B, F32 *C, bool AT, bool BT, int AR, int AC, int BR, int BC)
{
    matmul_(B, A, C, BT, AT, BC, BR, AC, AR);
}

void fc(Vec3<F32> input, Vec2<F32> kernel, Vec3<F32> output)
{
    matmul(input.data, kernel.data, output.data, false, true, input.z * input.y, input.x, kernel.y, kernel.x);
}

template <typename T>
void fc(Vec3<T> input, Vec2<F32> kernel, Vec3<F32> output)
{
    auto temp = init<F32>(input.z, input.y, input.x);
    copy(input, temp);
    fc(temp, kernel, output);
    deinit(temp);
}
}  // namespace CUDA

void cuda_fc(Dyn3 input, Dyn2 kernel, Dyn3 output) { IfType(T, input.type, CUDA::fc<T>(input, kernel, output)); }
}  // namespace Core
}  // namespace Spyker
