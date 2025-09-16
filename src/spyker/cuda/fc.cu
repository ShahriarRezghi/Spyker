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

namespace Spyker
{
namespace Core
{
namespace CUDA
{
void _matmul(F32 *A, F32 *B, F32 *C, bool AT, bool BT, int AR, int AC, int BR, int BC)
{
    if (!cublas_static) cublas_static = std::unique_ptr<cublas>(new cublas);

    F32 alpha = 1, beta = 0;
    int ATR = AT ? AC : AR, ATC = AT ? AR : AC, BTC = BT ? BR : BC;
    cublasOperation_t AT_ = AT ? CUBLAS_OP_T : CUBLAS_OP_N, BT_ = BT ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(cublas_static->handle, AT_, BT_, ATR, BTC, ATC, &alpha, A, AR, B, BR, &beta, C, ATR);
}
void _matmul(F64 *A, F64 *B, F64 *C, bool AT, bool BT, int AR, int AC, int BR, int BC)
{
    if (!cublas_static) cublas_static = std::unique_ptr<cublas>(new cublas);

    F64 alpha = 1, beta = 0;
    int ATR = AT ? AC : AR, ATC = AT ? AR : AC, BTC = BT ? BR : BC;
    cublasOperation_t AT_ = AT ? CUBLAS_OP_T : CUBLAS_OP_N, BT_ = BT ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasDgemm(cublas_static->handle, AT_, BT_, ATR, BTC, ATC, &alpha, A, AR, B, BR, &beta, C, ATR);
}

template <typename T>
void matmul(T *A, T *B, T *C, bool AT, bool BT, int AR, int AC, int BR, int BC)
{
    _matmul(B, A, C, BT, AT, BC, BR, AC, AR);
}

template <typename T>
void _fc(Vec3<T> input, Vec2<T> kernel, Vec3<T> output)
{
    matmul(input.data, kernel.data, output.data, false, true, input.z * input.y, input.x, kernel.y, kernel.x);
}
void _fc(Vec3<C16> input, Vec2<C16> kernel, Vec3<C16> output)
{
    SpykerAssert(false, "CUDA::FC", "F16 is not supported with BLAS.");
}

template <typename T>
void fc(Vec3<T> input, Vec2<T> kernel, Vec3<T> output)
{
    auto temp = init<T>(input.z, input.y, input.x);
    copy(input, temp);
    _fc(temp, kernel, output);
    deinit(temp);
}
}  // namespace CUDA

void cuda_fc(Dyn3 input, Dyn2 kernel, Dyn3 output) { IfReal(T, kernel.type, CUDA::fc<T>(input, kernel, output)); }
}  // namespace Core
}  // namespace Spyker
